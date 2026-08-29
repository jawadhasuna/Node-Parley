// Stage C: the Node-Parley spectrum-sharing problem, in ns-3.
//
// Stages A, B and B+ ran on an environment I wrote, where "two nodes on one
// channel" meant a hard rule: both fail. That rule was an assumption. ns-3
// does not assume it -- it simulates real 802.11 CSMA/CA, so two links sharing
// a channel back off, contend, and each get roughly HALF the airtime rather
// than nothing.
//
// That difference is the point of this stage. If the optimal policy comes out
// the same, the simplification was harmless. If it comes out different, the
// Gym results were shaped by my assumption rather than by the problem.
//
// STRUCTURE
// ---------
// N transmitter/receiver pairs (links) share M non-overlapping 5 GHz WiFi
// channels. Every step:
//
//   1. each link picks a channel, epsilon-greedy from its own Q-table
//   2. a short ns-3 simulation runs with that assignment
//   3. throughput per link is measured from bytes actually received
//   4. each link updates its Q-table
//
// Each step is an independent simulation rather than one long run with
// mid-flight channel switching. Switching a live WifiPhy is possible but adds
// failure modes; independent runs are slower and much easier to trust.
//
// The Q-learning is the same update rule as qlearning.py, written out again
// rather than bridged from Python. ns3-gym and ns3-ai exist for that, but they
// track ns-3 releases poorly and would make this stage depend on a third
// party staying current. Thirty lines of C++ does not.
//
// Build:  copy to ns-3's scratch/ then ./ns3 build
// Run:    ./ns3 run "spectrum-qlearning --episodes=400"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace ns3;

// Matches the Python experiments: 3 links, 4 channels.
static const uint32_t N_LINKS = 3;
static const uint32_t N_CHANNELS = 4;

// Non-overlapping 5 GHz channels. Non-overlapping matters: adjacent 2.4 GHz
// channels bleed into each other, which would add a second effect on top of
// the one being measured.
static const uint32_t CHANNEL_NUMBERS[N_CHANNELS] = {36, 40, 44, 48};

// Channel quality, mirroring the Python environment's unequal channels. Here
// it is expressed as distance: a longer link has lower SNR and so a lower
// rate, which is the physical version of "channel 0 is the best one".
static const double LINK_DISTANCE[N_CHANNELS] = {5.0, 15.0, 30.0, 45.0};

struct StepResult
{
    std::vector<double> throughputMbps; // per link
    uint32_t sharedChannels;            // channels carrying more than one link
};

/**
 * Run one short simulation with a fixed channel assignment.
 *
 * Every link is a transmitter and a receiver saturating a UDP flow. Links
 * assigned the same channel are within carrier-sense range of each other, so
 * 802.11 makes them contend for airtime.
 */
StepResult
RunStep(const std::vector<uint32_t>& channels, double duration, uint32_t seed)
{
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(seed);

    NodeContainer txNodes;
    NodeContainer rxNodes;
    txNodes.Create(N_LINKS);
    rxNodes.Create(N_LINKS);

    // Place every node close enough that same-channel links hear each other.
    // If they were far apart they would not contend, and the whole question
    // of sharing would disappear.
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positions = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < N_LINKS; ++i)
    {
        positions->Add(Vector(i * 2.0, 0.0, 0.0)); // transmitter
    }
    for (uint32_t i = 0; i < N_LINKS; ++i)
    {
        // Receiver distance depends on the CHOSEN channel, which is how
        // channel quality enters: a link on a poor channel is effectively a
        // longer link and negotiates a lower rate.
        positions->Add(Vector(i * 2.0, LINK_DISTANCE[channels[i]], 0.0));
    }
    mobility.SetPositionAllocator(positions);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(txNodes);
    mobility.Install(rxNodes);

    InternetStackHelper stack;
    stack.Install(txNodes);
    stack.Install(rxNodes);

    ApplicationContainer sinks;
    std::vector<Ptr<PacketSink>> sinkPtrs;

    // ONE channel object, shared by every link. This is load-bearing: giving
    // each link its own YansWifiChannel puts them in separate universes that
    // cannot hear each other, so setting channel NUMBERS has no effect and
    // nothing ever contends. The first version of this file did exactly that
    // and reported that sharing a channel costs almost nothing -- which was
    // not a finding about CSMA/CA, it was three isolated links.
    //
    // With one channel, every PHY receives every transmission and decides for
    // itself whether the signal is on its operating channel. Same channel
    // number means real interference; different numbers mean none.
    YansWifiChannelHelper chanHelper = YansWifiChannelHelper::Default();
    Ptr<YansWifiChannel> sharedChannel = chanHelper.Create();

    // One WiFi network per link, each on its chosen channel. Links sharing a
    // channel number share the medium and must contend.
    for (uint32_t i = 0; i < N_LINKS; ++i)
    {
        YansWifiPhyHelper phy;
        phy.SetChannel(sharedChannel);

        std::ostringstream chanSetting;
        chanSetting << "{" << CHANNEL_NUMBERS[channels[i]]
                    << ", 20, BAND_5GHZ, 0}";
        phy.Set("ChannelSettings", StringValue(chanSetting.str()));

        WifiHelper wifi;
        wifi.SetStandard(WIFI_STANDARD_80211a);
        // Minstrel adapts the rate to the link, so a poor channel yields
        // lower throughput on its own rather than by a hardcoded penalty.
        wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");

        WifiMacHelper mac;
        mac.SetType("ns3::AdhocWifiMac");

        NodeContainer pair(txNodes.Get(i), rxNodes.Get(i));
        NetDeviceContainer devices = wifi.Install(phy, mac, pair);

        Ipv4AddressHelper address;
        std::ostringstream subnet;
        subnet << "10.1." << (i + 1) << ".0";
        address.SetBase(subnet.str().c_str(), "255.255.255.0");
        Ipv4InterfaceContainer ifaces = address.Assign(devices);

        uint16_t port = 9000 + i;
        PacketSinkHelper sinkHelper(
            "ns3::UdpSocketFactory",
            InetSocketAddress(Ipv4Address::GetAny(), port));
        ApplicationContainer sinkApp = sinkHelper.Install(rxNodes.Get(i));
        sinkApp.Start(Seconds(0.0));
        sinkPtrs.push_back(DynamicCast<PacketSink>(sinkApp.Get(0)));

        // Saturating traffic: the link always has something to send, so
        // measured throughput reflects what the channel allowed rather than
        // what the application happened to offer.
        OnOffHelper onoff("ns3::UdpSocketFactory",
                          InetSocketAddress(ifaces.GetAddress(1), port));
        onoff.SetAttribute("OnTime",
                           StringValue("ns3::ConstantRandomVariable[Constant=1]"));
        onoff.SetAttribute("OffTime",
                           StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        onoff.SetAttribute("DataRate", StringValue("54Mbps"));
        onoff.SetAttribute("PacketSize", UintegerValue(1400));
        ApplicationContainer app = onoff.Install(txNodes.Get(i));
        app.Start(Seconds(0.1));
        app.Stop(Seconds(duration));
    }

    Simulator::Stop(Seconds(duration + 0.1));
    Simulator::Run();

    StepResult result;
    result.throughputMbps.resize(N_LINKS);
    for (uint32_t i = 0; i < N_LINKS; ++i)
    {
        double bytes = static_cast<double>(sinkPtrs[i]->GetTotalRx());
        result.throughputMbps[i] = bytes * 8.0 / (duration * 1e6);
    }

    std::vector<uint32_t> counts(N_CHANNELS, 0);
    for (uint32_t c : channels)
    {
        counts[c]++;
    }
    result.sharedChannels =
        std::count_if(counts.begin(), counts.end(), [](uint32_t n) { return n > 1; });

    Simulator::Destroy();
    return result;
}

/// Encode which channels were occupied last step, matching qlearning.py.
uint32_t
EncodeState(const std::vector<uint32_t>& channels)
{
    uint32_t occupancy = 0;
    for (uint32_t c : channels)
    {
        occupancy |= (1u << c);
    }
    return occupancy;
}

int
main(int argc, char* argv[])
{
    uint32_t episodes = 300;
    double stepDuration = 0.3;
    std::string rewardMode = "collaborative";
    double alpha = 0.1;
    double gamma = 0.5;
    uint32_t seed = 1;

    CommandLine cmd(__FILE__);
    cmd.AddValue("episodes", "training steps", episodes);
    cmd.AddValue("duration", "simulated seconds per step", stepDuration);
    cmd.AddValue("reward", "selfish | collaborative", rewardMode);
    cmd.AddValue("alpha", "learning rate", alpha);
    cmd.AddValue("gamma", "discount factor", gamma);
    cmd.AddValue("seed", "random seed", seed);
    cmd.Parse(argc, argv);

    const uint32_t nStates = 1u << N_CHANNELS;
    // One table per link. Separate tables are what let the links specialise --
    // the RLlib run in Stage B2 showed a SHARED policy cannot, because
    // identical inputs force identical actions.
    std::vector<std::vector<std::vector<double>>> q(
        N_LINKS,
        std::vector<std::vector<double>>(nStates, std::vector<double>(N_CHANNELS, 0.0)));

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::uniform_int_distribution<uint32_t> randomChannel(0, N_CHANNELS - 1);

    std::vector<uint32_t> channels(N_LINKS);
    for (uint32_t i = 0; i < N_LINKS; ++i)
    {
        channels[i] = randomChannel(rng);
    }
    uint32_t state = EncodeState(channels);

    std::cout << "ns-3 " << N_LINKS << " links, " << N_CHANNELS << " channels\n"
              << "reward: " << rewardMode << ", " << episodes << " steps of "
              << stepDuration << " s\n\n"
              << std::setw(8) << "step" << std::setw(14) << "throughput"
              << std::setw(10) << "shared" << std::setw(10) << "epsilon" << "\n"
              << std::string(42, '-') << "\n";

    std::vector<double> recentThroughput;
    std::vector<double> history;

    for (uint32_t ep = 0; ep < episodes; ++ep)
    {
        double epsilon = std::max(0.02, 1.0 - static_cast<double>(ep) / (episodes * 0.5));

        std::vector<uint32_t> actions(N_LINKS);
        for (uint32_t i = 0; i < N_LINKS; ++i)
        {
            if (uniform(rng) < epsilon)
            {
                actions[i] = randomChannel(rng);
            }
            else
            {
                const std::vector<double>& row = q[i][state];
                double best = *std::max_element(row.begin(), row.end());
                // Break ties at random, or an all-zero table always returns
                // channel 0 and early exploration is biased.
                std::vector<uint32_t> bestActions;
                for (uint32_t a = 0; a < N_CHANNELS; ++a)
                {
                    if (row[a] >= best - 1e-12)
                    {
                        bestActions.push_back(a);
                    }
                }
                actions[i] = bestActions[rng() % bestActions.size()];
            }
        }

        StepResult res = RunStep(actions, stepDuration, seed + ep);
        double total =
            std::accumulate(res.throughputMbps.begin(), res.throughputMbps.end(), 0.0);

        uint32_t nextState = EncodeState(actions);
        for (uint32_t i = 0; i < N_LINKS; ++i)
        {
            // Same two rules as the Python experiments: paid for your own
            // throughput, or paid the ensemble average as SC2 scored teams.
            double reward = (rewardMode == "collaborative")
                                ? total / N_LINKS
                                : res.throughputMbps[i];

            double bestNext = *std::max_element(q[i][nextState].begin(),
                                                q[i][nextState].end());
            double tdError = reward + gamma * bestNext - q[i][state][actions[i]];
            q[i][state][actions[i]] += alpha * tdError;
        }

        state = nextState;
        channels = actions;
        history.push_back(total);
        recentThroughput.push_back(total);
        if (recentThroughput.size() > 20)
        {
            recentThroughput.erase(recentThroughput.begin());
        }

        if (ep % 25 == 0 || ep == episodes - 1)
        {
            double avg = std::accumulate(recentThroughput.begin(),
                                         recentThroughput.end(), 0.0) /
                         recentThroughput.size();
            std::cout << std::setw(8) << ep << std::setw(11) << std::fixed
                      << std::setprecision(2) << avg << " Mbps" << std::setw(10)
                      << res.sharedChannels << std::setw(10)
                      << std::setprecision(3) << epsilon << "\n";
        }
    }

    // --- greedy evaluation ---------------------------------------------------
    std::cout << "\ngreedy evaluation (20 steps)\n";
    double evalTotal = 0.0;
    uint32_t evalShared = 0;
    std::vector<uint32_t> chanUse(N_CHANNELS, 0);

    for (uint32_t ep = 0; ep < 20; ++ep)
    {
        std::vector<uint32_t> actions(N_LINKS);
        for (uint32_t i = 0; i < N_LINKS; ++i)
        {
            const std::vector<double>& row = q[i][state];
            actions[i] = std::distance(row.begin(),
                                       std::max_element(row.begin(), row.end()));
            chanUse[actions[i]]++;
        }
        StepResult res = RunStep(actions, stepDuration, 90000 + ep);
        evalTotal +=
            std::accumulate(res.throughputMbps.begin(), res.throughputMbps.end(), 0.0);
        evalShared += res.sharedChannels;
        state = EncodeState(actions);
    }

    std::cout << "  throughput      : " << std::fixed << std::setprecision(2)
              << evalTotal / 20 << " Mbps\n"
              << "  shared channels : " << std::setprecision(3)
              << static_cast<double>(evalShared) / 20 << " per step\n"
              << "  channel use     : ";
    for (uint32_t c = 0; c < N_CHANNELS; ++c)
    {
        std::cout << chanUse[c] << " ";
    }
    std::cout << "\n";

    return 0;
}

# Stage C: ns-3

`spectrum-qlearning.cc` runs the Node-Parley spectrum-sharing problem in
ns-3.48 with real 802.11 — CSMA/CA, backoff, propagation loss, and Minstrel
rate control — instead of a collision rule written by hand.

Only the source lives here. The ns-3 build tree is several GB of object files
and belongs nowhere near git.

## Build

ns-3 is Linux-native; on Windows the supported route is WSL2.

```bash
wsl -d Ubuntu-22.04
sudo apt update && sudo apt install -y cmake ninja-build ccache python3-pip
```

**Ubuntu 22.04 ships CMake 3.22 and ns-3.48 requires 3.25+.** Install a newer
one into your user directory rather than adding a third-party APT repository:

```bash
pip install --user cmake
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
cmake --version          # needs to be >= 3.25
```

Then ns-3 itself. **Build in the Linux home directory, not under `/mnt/c`** —
WSL2 reaching into the Windows filesystem is roughly 10x slower for the
many-small-files access a compiler does, which turns a 15-minute build into
over an hour.

```bash
cd ~
wget https://www.nsnam.org/releases/ns-allinone-3.48.tar.bz2
tar xjf ns-allinone-3.48.tar.bz2
cd ns-allinone-3.48/ns-3.48

./ns3 configure -d optimized \
  --enable-modules=core,network,internet,wifi,applications,mobility,propagation,spectrum,stats,flow-monitor \
  --disable-examples --disable-tests
./ns3 build
```

`-d optimized` matters: the default debug build compiles in assertions and
logging and runs several times slower, and this scenario runs hundreds of
simulations. Restricting modules and skipping examples and tests roughly
halves the build.

## Run

```bash
cp /path/to/Node-Parley/ns3/spectrum-qlearning.cc ~/ns-allinone-3.48/ns-3.48/scratch/
cd ~/ns-allinone-3.48/ns-3.48
./ns3 build spectrum-qlearning

./ns3 run "spectrum-qlearning --episodes=300 --duration=0.3 --reward=collaborative"
./ns3 run "spectrum-qlearning --episodes=300 --duration=0.3 --reward=selfish"
```

Options: `--episodes`, `--duration` (simulated seconds per step), `--reward`
(`selfish` or `collaborative`), `--alpha`, `--gamma`, `--seed`.

## What it does

Three transmitter/receiver pairs share four non-overlapping 5 GHz channels.
Each step:

1. every link picks a channel, epsilon-greedy from its own Q-table
2. a short ns-3 simulation runs with that assignment
3. throughput per link is measured from bytes actually received
4. every link updates its Q-table

Each step is an independent simulation rather than one long run with mid-flight
channel switching. Switching a live `WifiPhy` is possible but adds failure
modes; independent runs are slower and much easier to trust.

Channel quality is physical rather than a hardcoded weight: a worse channel
means a longer link, so Minstrel negotiates a lower rate on its own.

Separate Q-tables per link, deliberately. Stage B2 showed a *shared* policy
cannot specialise, because identical observations force identical actions.

## Results

Both reward rules converge to the same policy: three links on three distinct
channels, zero contention, 61.03 Mbps against ~41 for random assignment.

See the main README for what that means alongside Stages B and B+.

## Two things worth knowing before extending this

**Sharing must cost something.** An early version gave each link its own
`YansWifiChannel`, which put them in separate universes that could not hear
each other. Channel numbers were set on radios never on the same medium, so
nothing contended and sharing appeared free. If a change makes the `shared`
column stop correlating with throughput, that is the first thing to check.

**No MCS in the action space.** Minstrel adapts the rate automatically, so the
node chooses only a channel. Stage B+ found the reward rules diverge only when
MCS is a decision, which is why Stage C shows no difference — and it is the
most interesting way to extend this.

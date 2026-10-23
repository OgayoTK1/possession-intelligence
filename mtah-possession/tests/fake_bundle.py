"""Write a SYNTHETIC match bundle in the competition's documented file layout.

Contains no competition data: positions are random walks (or, if available, an open Metrica
match), relabelled onto 46 fragmenting track slots with x,y,speed channels, padded to the
requested number of frames. Used to exercise the loader -> features -> model -> CSV -> official
validator path end to end where the real bundle is not available.

    python tests/fake_bundle.py OUT_DIR [--frames 199319]
"""

import argparse
import os

import numpy as np

HEAD = """<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig><FrameRate>25.0</FrameRate><ProviderName>Metrica Sports</ProviderName></GlobalConfig>
    <Sessions><Session id="s1"><FieldSize><Width>105</Width><Height>68</Height></FieldSize></Session></Sessions>
    <Teams>
      <Team id="T_H"><Name>Home FC</Name></Team>
      <Team id="T_A"><Name>Away FC</Name></Team>
    </Teams>
    <Players>
{players}
    </Players>
    <PlayerChannels>
{channels}
    </PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification endFrame="{end}" separator=":" startFrame="1">
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
{registers}
      </SplitRegister>
      <SplitRegister separator=",">
        <BallChannelRef channelId="x"/>
        <BallChannelRef channelId="y"/>
        <BallChannelRef channelId="s"/>
      </SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
"""

N_SLOTS = 46


def write_bundle(out_dir, n_frames=199319, seed=0):
    rng = np.random.RandomState(seed)
    folder = os.path.join(out_dir, "tactical")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, "BAR-ATH_tactical")

    players, channels, registers = [], [], []
    for k in range(N_SLOTS):
        tid = "T_H" if k < 24 else "T_A"
        pid = "P{}".format(k + 1)
        players.append('      <Player id="{}" teamId="{}"><Name>Track {}</Name>'
                       '<ShirtNumber>{}</ShirtNumber></Player>'.format(pid, tid, k + 1, k + 1))
        for ch in ("x", "y", "s"):
            channels.append('      <PlayerChannel id="{0}_{1}" playerId="{0}" channelId="{1}"/>'
                            .format(pid, ch))
        registers.append('        <SplitRegister separator=",">' + "".join(
            '<PlayerChannelRef playerChannelId="{}_{}"/>'.format(pid, ch)
            for ch in ("x", "y", "s")) + "</SplitRegister>")
    with open(stem + "_atd.xml", "w") as f:
        f.write(HEAD.format(players="\n".join(players), channels="\n".join(channels),
                            registers="\n".join(registers), end=n_frames))

    # 22 random-walk players mapped onto 46 slots that hand over every ~60 s
    pos = rng.rand(22, 2)
    ball = np.array([0.5, 0.5])
    with open(stem + "_atd.txt", "w") as f:
        for i in range(n_frames):
            pos = np.clip(pos + rng.normal(0, 0.0015, pos.shape), 0.01, 0.99)
            ball = np.clip(ball + rng.normal(0, 0.004, 2), -0.02, 1.02)
            epoch = i // 1500
            groups = ["NaN,NaN,NaN"] * N_SLOTS
            for p in range(22):
                team_off = 0 if p < 11 else N_SLOTS // 2
                slot = team_off + (p % 11 + epoch) % (N_SLOTS // 2)
                if rng.rand() < 0.07:
                    continue
                groups[slot] = "{:.5f},{:.5f},{:.2f}".format(pos[p, 0], pos[p, 1], rng.rand() * 5)
            b = "NaN,NaN,NaN" if (i // 500) % 5 == 2 else "{:.5f},{:.5f},{:.2f}".format(
                ball[0], ball[1], rng.rand() * 10)
            f.write("{}:{}:{}\n".format(i + 1, ";".join(groups), b))

    with open(stem + "_smart_tagging.xml", "w") as f:
        f.write('<file><ALL_INSTANCES><instance><ID>1</ID><start>447.0</start><end>452.0</end>'
                '<code>SET PIECES</code><label><group>Team</group><text>Home FC</text></label>'
                '<label><group>Type</group><text>Kick Off (Start)</text></label></instance>'
                '</ALL_INSTANCES></file>')
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=199319)
    a = ap.parse_args()
    write_bundle(a.out, a.frames)
    print("wrote synthetic bundle to", a.out)

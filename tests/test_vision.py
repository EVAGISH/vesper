"""The vision policy consumes exactly what SearchEnv emits, and checkpoints carry their width."""
import torch

from vesper.lab.ppo import ActorCritic, RunningNorm, load_policy
from vesper.lab.search_task import PROPRIO_DIM
from vesper.lab.vision import VisionActorCritic


def test_vision_policy_matches_the_env_observation_contract():
    n, k, g, res = 4, 3, 8, 128
    priv_dim = 12 + 8 * k + g * g + 3
    net = VisionActorCritic(PROPRIO_DIM, 3, priv_dim=priv_dim, res=res)
    pixels = torch.randint(0, 256, (n, res, res, 3), dtype=torch.uint8)   # env hands out uint8 NHWC
    proprio = torch.randn(n, PROPRIO_DIM)
    priv = torch.randn(n, priv_dim)
    h = net.initial_state(n)
    mean, value, h2, aux = net(pixels, proprio, h, priv=priv, done=torch.tensor([False, True, False, False]))
    assert mean.shape == (n, 3) and value.shape == (n,) and h2.shape == (n, 512) and aux.shape == (n, 3)
    assert torch.isfinite(mean).all() and torch.isfinite(value).all()
    a = net.dist(mean).sample()
    assert a.shape == (n, 3)
    # "a couple of million parameters", not a toy
    assert 2_000_000 < net.n_params() < 6_000_000, net.n_params()


def test_checkpoint_carries_the_network_width(tmp_path):
    ac = ActorCritic(20, 3, hidden=(512, 512, 256))
    norm = RunningNorm(20)
    ck = {"ac": ac.state_dict(), "norm": norm.state_dict(), "obs_dim": 20, "act_dim": 3,
          "hidden": list(ac.hidden)}
    torch.save(ck, tmp_path / "p.pt")
    ac2, norm2 = load_policy(torch.load(tmp_path / "p.pt"))
    assert ac2.hidden == (512, 512, 256)
    x = torch.randn(2, 20)
    assert torch.allclose(ac2.actor(x), ac.actor(x))
    # an old checkpoint without the field is the class default
    old = {k: v for k, v in ck.items() if k != "hidden"}
    old["ac"] = ActorCritic(20, 3).state_dict()
    ac3, _ = load_policy(old)
    assert ac3.hidden == (256, 256, 128)

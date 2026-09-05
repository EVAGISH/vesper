"""The vision policy consumes exactly what ChaseEnv emits, at the size the airframe can run."""
import torch

from vesper.lab.frames import PROPRIO_DIM
from vesper.lab.ppo import ActorCritic, RunningNorm, load_policy
from vesper.lab.vision import VisionActorCritic


def test_vision_policy_matches_the_env_observation_contract():
    n, k, res = 4, 6, 96
    priv_dim = 12 + 6 * k + 1
    net = VisionActorCritic(PROPRIO_DIM, 3, priv_dim=priv_dim, res=res)
    pixels = torch.randint(0, 256, (n, res, res, 3), dtype=torch.uint8)   # uint8 NHWC from the env
    depth = torch.rand(n, res, res, 1)                                     # [0,1] from DepthModel
    proprio = torch.randn(n, PROPRIO_DIM)
    priv = torch.randn(n, priv_dim)
    h = net.initial_state(n)
    mean, value, belief, h2 = net(pixels, depth, proprio, h, priv=priv,
                                  done=torch.tensor([False, True, False, False]))
    assert mean.shape == (n, 3) and value.shape == (n,) and h2.shape == (n, 256) and belief.shape == (n, 3)
    assert torch.isfinite(mean).all() and torch.isfinite(value).all()
    assert net.dist(mean).sample().shape == (n, 3)
    # sized for a small drone computer: ~1.4M parameters on board, ~17M MACs a frame
    on_board = net.n_params(deployed=True)
    assert 1_000_000 < on_board < 2_000_000, on_board
    assert 10_000_000 < net.macs_per_frame() < 30_000_000, net.macs_per_frame()


def test_memory_is_zeroed_on_done_and_carried_otherwise():
    net = VisionActorCritic(PROPRIO_DIM, 3, res=96)
    n = 2
    px = torch.randint(0, 256, (n, 96, 96, 3), dtype=torch.uint8)
    dp = torch.rand(n, 96, 96, 1)
    pr = torch.randn(n, PROPRIO_DIM)
    h = torch.randn(n, 256)
    _, _, _, h_reset = net(px, dp, pr, h, done=torch.tensor([True, True]))
    _, _, _, h_zero = net(px, dp, pr, torch.zeros(n, 256))
    assert torch.allclose(h_reset, h_zero)
    _, _, _, h_keep = net(px, dp, pr, h, done=torch.tensor([False, False]))
    assert not torch.allclose(h_keep, h_zero)


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
    old = {k: v for k, v in ck.items() if k != "hidden"}
    old["ac"] = ActorCritic(20, 3).state_dict()
    ac3, _ = load_policy(old)
    assert ac3.hidden == (256, 256, 128)


def test_deployed_path_needs_no_privileged_vector():
    """On the drone there is no critic input: the actor still runs."""
    net = VisionActorCritic(PROPRIO_DIM, 3, priv_dim=49, res=96)
    n = 2
    px = torch.randint(0, 256, (n, 96, 96, 3), dtype=torch.uint8)
    dp = torch.rand(n, 96, 96, 1)
    pr = torch.randn(n, PROPRIO_DIM)
    h = net.initial_state(n)
    mean, value, belief, h2 = net(px, dp, pr, h)          # priv=None
    assert mean.shape == (n, 3) and torch.isfinite(mean).all()
    assert value.shape == (n,) and float(value.abs().max()) == 0.0
    a, h3 = net.act(px, dp, pr, h)
    assert torch.allclose(a, mean) and torch.allclose(h3, h2)

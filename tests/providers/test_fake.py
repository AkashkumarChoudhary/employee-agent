import pytest

from employee_agent.providers.base import ProviderError
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import VerifierVerdict


async def test_generate_structured_returns_registered_instance():
    want = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")
    p = FakeProvider(responses={VerifierVerdict: want})
    got = await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)
    assert got == want


async def test_embed_is_deterministic_and_right_shape():
    p = FakeProvider(embed_dim=8)
    a = await p.embed(["hello", "world"])
    b = await p.embed(["hello", "world"])
    assert len(a) == 2 and len(a[0]) == 8
    assert a == b  # deterministic


async def test_fail_mode_raises():
    p = FakeProvider(fail=True)
    with pytest.raises(ProviderError):
        await p.embed(["x"])
    with pytest.raises(ProviderError):
        await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)

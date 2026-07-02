import pytest

from employee_agent.providers.base import ProviderError
from employee_agent.providers.failover import FailoverProvider
from employee_agent.providers.fake import FakeProvider
from employee_agent.schemas import VerifierVerdict


async def test_uses_primary_when_healthy():
    want = VerifierVerdict(grounded=True, unsupported_claims=[], action="accept")
    p = FailoverProvider(FakeProvider(responses={VerifierVerdict: want}),
                         FakeProvider(fail=True))
    got = await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)
    assert got == want


async def test_falls_back_when_primary_fails():
    want = VerifierVerdict(grounded=False, unsupported_claims=["x"], action="accept")
    p = FailoverProvider(FakeProvider(fail=True),
                         FakeProvider(responses={VerifierVerdict: want}))
    got = await p.generate_structured(system="s", prompt="p", schema=VerifierVerdict)
    assert got == want


async def test_raises_when_both_fail():
    p = FailoverProvider(FakeProvider(fail=True), FakeProvider(fail=True))
    with pytest.raises(ProviderError):
        await p.embed(["x"])


async def test_name_composes():
    p = FailoverProvider(FakeProvider(), FakeProvider())
    assert p.name == "fake->fake"

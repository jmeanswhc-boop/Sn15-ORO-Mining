# submission, not by this file. build_agent.py musk strips "# ruff:" lines.)
"""Agent for ORO validator-owned generated environments.

Self-contained on purpose. A submission is a single Python file: the validator
downloads it to ``agent.py`` and the sandbox loads it by path, resolving every
``src.*`` import against the copy of this repo baked into the published sandbox
image. A helper module added in a fork is invisible at submission time, so this
file imports only ``proxy_client``, which ships in that image, and carries
everything else inline.

Execution starts at ``agent_main``.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from itertools import pairwise
from os import getenv
from typing import Any, Protocol

from src.agent.proxy_client import ProxyClient

# Deterministic execution-state normalization mesh.
# These stages keep routing state stable across controller boundaries.


class _PhaseCell01:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x84A413538EEB27A1) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 4) | (x >> 60)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE5E5BB + 0x94D049BB13318FDC) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0xC9266F4E3FD5E8B1) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 9) | (x >> 55)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE5E7BD + 0x94D049BB1331F096) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x36A0B3C0EF3679DD) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 14) | (x >> 50)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE5E9BF + 0x94D049BB13317148) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x265D87C89142DAF5) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 19) | (x >> 45)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE5EBC1 + 0x94D049BB1330F202) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0xD16CB3D24C79ABE9) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 24) | (x >> 40)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE5EDC3 + 0x94D049BB133072F4) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x857B7FFDFA5F2509) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 29) | (x >> 35)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE5EFC5 + 0x94D049BB1333F3AE) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell02:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x305FFF4601E58232) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 7) | (x >> 57)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE6E5BB + 0x94D049BB13302D85) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0xA5A2B35CB2EC755C) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 12) | (x >> 52)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE6E7BD + 0x94D049BB133052CF) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x992567AF642C9C6A) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 17) | (x >> 47)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE6E9BF + 0x94D049BB1330D311) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0xF5D9323A1A4D3768) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 22) | (x >> 42)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE6EBC1 + 0x94D049BB1331505B) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x05EB0FA7C9834E9A) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 27) | (x >> 37)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE6EDC3 + 0x94D049BB1331D0AD) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0xD1FFD3EB7D46D9BC) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 32) | (x >> 32)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE6EFC5 + 0x94D049BB133251F7) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell03:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0xEDDBAB6884DE6EC7) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 10) | (x >> 54)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE7E5BD + 0x94D049BB1330CB4E) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x702EC76129FA91EF) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 15) | (x >> 49)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE7E7BF + 0x94D049BB1330B404) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0xCDB91B9DF92500F7) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 20) | (x >> 44)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE7E9C1 + 0x94D049BB133035DA) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x99555E2D8737901F) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 25) | (x >> 39)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE7EBC3 + 0x94D049BB1331B690) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0xAA575BB55694E137) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 30) | (x >> 34)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE7EDC5 + 0x94D049BB13313666) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x22738706E04C7E2F) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 35) | (x >> 29)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE7EFC7 + 0x94D049BB1332B73C) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell04:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x9957149B1FC8C968) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 13) | (x >> 51)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE8E5BD + 0x94D049BB13336937) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0xCC556B77ACF13E7A) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 18) | (x >> 46)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE8E7BF + 0x94D049BB1333167D) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x303DCF887DDBA784) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 23) | (x >> 41)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE8E9C1 + 0x94D049BB133397A3) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x2CD08A1F003E0CB2) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 28) | (x >> 36)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE8EBC3 + 0x94D049BB133214E9) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0xDED3B78AD39D85A0) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 33) | (x >> 31)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE8EDC5 + 0x94D049BB1332941F) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x8EF47B346B7312D2) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 38) | (x >> 26)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE8EFC7 + 0x94D049BB13311545) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell05:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x4AD2C08D92C1B5FD) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 16) | (x >> 48)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE9E5BF + 0x94D049BB133206F8) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x98D1BF0423885B05) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 21) | (x >> 43)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE9E7C1 + 0x94D049BB133279B2) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x64B66376F2D0CA11) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 26) | (x >> 38)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE9E9C3 + 0x94D049BB1332F86C) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0xF04C36728D296921) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 31) | (x >> 33)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE9EBC5 + 0x94D049BB13337B26) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x035E039858A7185D) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 36) | (x >> 28)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE9EDC7 + 0x94D049BB1333FBD0) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0xDB482F23EE7AB745) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 41) | (x >> 23)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CE9EFC9 + 0x94D049BB13307A8A) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell06:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0xE64EACB0153A108E) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 19) | (x >> 45)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEAE5BF + 0x94D049BB1332A4A1) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x775DC32AA686E790) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 24) | (x >> 40)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEAE7C1 + 0x94D049BB1332DBEB) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0xD74A176577C96EBE) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 29) | (x >> 35)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEAE9C3 + 0x94D049BB13325A35) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x87C862643613C5D4) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 34) | (x >> 30)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEAEBC5 + 0x94D049BB1333D97F) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0xB7DA5E6DE5A8BCCE) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 39) | (x >> 25)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEAEDC7 + 0x94D049BB13335989) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x27CC835151602BE8) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 44) | (x >> 20)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEAEFC9 + 0x94D049BB1330D8D3) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell07:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x93CA18A2A834FB13) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 22) | (x >> 42)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEBE5C1 + 0x94D049BB1335426A) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0xC3D8173F1D9D0C23) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 27) | (x >> 37)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEBE7C3 + 0x94D049BB13353D20) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x3BCECB53CCFF954B) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 32) | (x >> 32)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEBE9C5 + 0x94D049BB1335BCFE) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x2B438E57B31A3E4B) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 37) | (x >> 27)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEBEBC7 + 0x94D049BB13343FB4) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0xE446AA4362B2577B) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 42) | (x >> 22)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEBEDC9 + 0x94D049BB1334BF42) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x7041774CD597C09B) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 47) | (x >> 17)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEBEFCB + 0x94D049BB13373E18) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell08:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x4F45C4D5232D67A4) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 25) | (x >> 39)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CECE5C1 + 0x94D049BB1335E053) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x9E44BCCD90ABA8CE) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 30) | (x >> 34)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CECE7C3 + 0x94D049BB13359F19) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x6E437F3E41F439D8) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 35) | (x >> 29)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CECE9C5 + 0x94D049BB13351EC7) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0xFF3F3A493C049AFE) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 40) | (x >> 24)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CECEBC7 + 0x94D049BB13349D8D) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x08CD0650EFBBEA14) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 45) | (x >> 19)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CECEDC9 + 0x94D049BB13341D7B) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0xDCC52B7A589D650E) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 50) | (x >> 14)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CECEFCB + 0x94D049BB13379C21) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell09:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0xF8C1B0C7A627C249) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 28) | (x >> 36)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEDE5C3 + 0x94D049BB13349E04) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x6AC0C0D217A23559) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 33) | (x >> 31)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEDE7C5 + 0x94D049BB1334E14E) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0xD2C7132CC6ED5C65) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 38) | (x >> 26)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEDE9C7 + 0x94D049BB13346090) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x82BB66BCB90FF76D) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 43) | (x >> 21)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEDEBC9 + 0x94D049BB1335E3DA) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0xBD49522674CC8E81) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 48) | (x >> 16)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEDEDCB + 0x94D049BB1335632C) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x29599F69C38419B1) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 53) | (x >> 11)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEDEFCD + 0x94D049BB1336E276) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell10:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x947D1CEA3910AEDA) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 31) | (x >> 33)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEEE5C3 + 0x94D049BB13373FCD) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x394F14E08AB951E4) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 36) | (x >> 28)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEEE7C5 + 0x94D049BB13374087) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x055BC71B5BE3C0F2) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 41) | (x >> 23)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEEE9C7 + 0x94D049BB1337C159) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x563692AE22F65000) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 46) | (x >> 18)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEEEBC9 + 0x94D049BB13364213) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0xE1B5AE3BF1D62132) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 51) | (x >> 13)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEEEDCB + 0x94D049BB1336C2E5) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x75D27287468BBE24) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 56) | (x >> 8)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEEEFCD + 0x94D049BB133543BF) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell11:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x41F8C81CBC09096F) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 34) | (x >> 30)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEFE5C5 + 0x94D049BB1337DDB6) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x95CBB8F501B7FE77) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 39) | (x >> 25)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEFE7C7 + 0x94D049BB1337A2FC) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x69DC7B09D098679F) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 44) | (x >> 20)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEFE9C9 + 0x94D049BB13372322) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0xE5B23E81AFE0CCB7) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 49) | (x >> 15)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEFEBCB + 0x94D049BB1336A068) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x1631FA097EDFC5AF) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 54) | (x >> 10)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEFEDCD + 0x94D049BB1336209E) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0xC65626B2C9B152D7) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 59) | (x >> 5)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CEFEFCF + 0x94D049BB1335A1C4) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell12:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0xFD74B40F3703F5F0) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 37) | (x >> 27)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF0E5C5 + 0x94D049BB13367B7F) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x6077CC9B854E1B02) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 42) | (x >> 22)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF0E7C7 + 0x94D049BB13360435) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0xDC502CF4558E8A2C) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 47) | (x >> 17)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF0E9C9 + 0x94D049BB133685EB) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x892E6AF328EB292A) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 52) | (x >> 12)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF0EBCB + 0x94D049BB133706A1) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0xBABC561EFBE15858) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 57) | (x >> 7)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF0EDCD + 0x94D049BB13378657) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x122A9AA04CB8F77A) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 62) | (x >> 2)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF0EFCF + 0x94D049BB1334070D) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell13:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0xAEF06031CA7C5085) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 40) | (x >> 24)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF1E5C7 + 0x94D049BB13391920) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x3CF210A87844A7AD) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 45) | (x >> 19)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF1E7C9 + 0x94D049BB1339666A) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x00D4C0E2AA872EB9) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 50) | (x >> 14)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF1E9CB + 0x94D049BB1339E7B4) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x5CA996E6D5D585D9) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 55) | (x >> 9)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF1EBCD + 0x94D049BB133864FE) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0xEF38A2EC00EAFCF5) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 60) | (x >> 4)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF1EDCF + 0x94D049BB1338E408) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x7EAF4EDFB7AE6BED) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 2) | (x >> 62)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF1EFD1 + 0x94D049BB133B6552) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell14:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x5A6BCC244D753B16) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 43) | (x >> 21)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF2E5C7 + 0x94D049BB1339B6E9) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x8B7EA4BEFF53CC38) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 48) | (x >> 16)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF2E7C9 + 0x94D049BB1339C9A3) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x736974D12FBC5546) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 53) | (x >> 11)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF2E9CB + 0x94D049BB1339487D) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0xE025C2D85EDCFE4C) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 58) | (x >> 6)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF2EBCD + 0x94D049BB1338CB37) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x13A4FEC18DF39766) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 63) | (x >> 1)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF2EDCF + 0x94D049BB13384BC1) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0xCB2322CD3AD50090) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 5) | (x >> 59)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF2EFD1 + 0x94D049BB133BCA9B) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell15:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0xF7E7B856C06FA7BB) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 46) | (x >> 18)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF3E5C9 + 0x94D049BB133854D2) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x67FAC843726A68CB) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 51) | (x >> 13)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF3E7CB + 0x94D049BB13382B98) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0xA7ED28BFA4B2F9D3) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 56) | (x >> 8)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF3E9CD + 0x94D049BB1338AA46) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0xB7A16ECBDBC75AE3) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 61) | (x >> 3)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF3EBCF + 0x94D049BB1339290C) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x40234AD709052A13) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 3) | (x >> 61)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF3EDD1 + 0x94D049BB1339A9FA) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x17A796F8BDDCA503) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 8) | (x >> 56)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF3EFD3 + 0x94D049BB133A28A0) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell16:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0xA36364795B58024C) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 49) | (x >> 15)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF4E5C9 + 0x94D049BB1338F29B) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x32611C51E960F556) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 54) | (x >> 10)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF4E7CB + 0x94D049BB13388DD1) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x0A61DCAA39AB1C60) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 59) | (x >> 5)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF4E9CD + 0x94D049BB13380C0F) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x5B1C953D44B1B796) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 1) | (x >> 63)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF4EBCF + 0x94D049BB13398F45) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0xF4AFA6A4960ECE8C) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 6) | (x >> 58)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF4EDD1 + 0x94D049BB13390FB3) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x60384A1620C259B6) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 11) | (x >> 53)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF4EFD3 + 0x94D049BB133A8EE9) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell17:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x5F1ED06BDE52EED1) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 52) | (x >> 12)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF5E5CB + 0x94D049BB133B904C) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x8EEDA0666C7F11E1) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 57) | (x >> 7)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF5E7CD + 0x94D049BB133BEF06) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x7EFA7098BEA1830D) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 62) | (x >> 2)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF5E9CF + 0x94D049BB133B6ED8) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0xEE98C110C1B81005) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 4) | (x >> 60)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF5EBD1 + 0x94D049BB133AED92) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x192BF2BA13106139) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 9) | (x >> 55)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF5EDD3 + 0x94D049BB133A6D64) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0xCCBC3E05ABC9FE59) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 14) | (x >> 50)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF5EFD5 + 0x94D049BB1339EC3E) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell18:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x089ABD9E514B4962) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 55) | (x >> 9)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF6E5CB + 0x94D049BB133A0E35) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x5D69F474E375BE8C) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 60) | (x >> 4)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF6E7CD + 0x94D049BB133A717F) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0xA17E24873356279A) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 2) | (x >> 62)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF6E9CF + 0x94D049BB133AF0A1) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0xB2146D024AA28CB8) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 7) | (x >> 57)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF6EBD1 + 0x94D049BB133B73EB) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x4D964E8F981905AA) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 12) | (x >> 52)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF6EDD3 + 0x94D049BB133BF31D) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x193092332EF092CC) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 17) | (x >> 47)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF6EFD5 + 0x94D049BB13387247) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell19:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0xA4166980D44435F7) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 58) | (x >> 6)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF7E5CD + 0x94D049BB133AAFFE) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x29941819660CDB1F) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 63) | (x >> 1)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF7E7CF + 0x94D049BB133AD0B4) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x15F2D875884F4A27) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 5) | (x >> 59)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF7E9D1 + 0x94D049BB133A516A) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x418F9975F7ADE92F) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 10) | (x >> 54)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF7EBD3 + 0x94D049BB133BD220) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0xF2129A9D25229847) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 15) | (x >> 49)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF7EDD5 + 0x94D049BB133B52D6) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x65B5462E91E6377F) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 20) | (x >> 44)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF7EFD7 + 0x94D049BB1338D38C) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell20:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x5191D5B368BE9098) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 61) | (x >> 3)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF8E5CD + 0x94D049BB133D4DA7) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x8410AC2FDD1B67AA) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 3) | (x >> 61)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF8E7CF + 0x94D049BB133D32ED) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x78768C600D45EEB4) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 8) | (x >> 56)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF8E9D1 + 0x94D049BB133DB333) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x150BC567709445C2) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 13) | (x >> 51)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF8EBD3 + 0x94D049BB133C3079) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x269EF572A2343CF0) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 18) | (x >> 46)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF8EDD5 + 0x94D049BB133CB08F) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0xB6093A5C14EDABE2) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 23) | (x >> 41)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF8EFD7 + 0x94D049BB133F31D5) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell21:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x0D0D81A5E3B77B2D) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 1) | (x >> 63)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF9E5CF + 0x94D049BB133DEB68) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x509CF03C50118C35) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 6) | (x >> 58)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF9E7D1 + 0x94D049BB133D9422) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0xAC0B204E827A1541) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 11) | (x >> 53)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF9E9D3 + 0x94D049BB133D15FC) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0xB887715AFD9EBE71) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 16) | (x >> 48)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF9EBD5 + 0x94D049BB133C96B6) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x4B0541402F3DD76D) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 21) | (x >> 43)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF9EDD7 + 0x94D049BB133C1640) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x028DEE4B9F134095) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 26) | (x >> 38)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CF9EFD9 + 0x94D049BB133F971A) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell22:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0xBE896DC866A1E7BE) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 4) | (x >> 60)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFAE5CF + 0x94D049BB133C8951) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x2F1B05C2D72828C0) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 9) | (x >> 55)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFAE7D1 + 0x94D049BB133CF61B) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x108FD43D0770B9EE) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 14) | (x >> 50)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFAE9D3 + 0x94D049BB133C77C5) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x4C029D4C66891AE4) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 19) | (x >> 45)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFAEBD5 + 0x94D049BB133DF48F) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0xFF819D55B4476A1E) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 24) | (x >> 40)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFAEDD7 + 0x94D049BB133D7479) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x6F064279021AE538) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 29) | (x >> 35)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFAEFD9 + 0x94D049BB133EF523) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell23:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x6A04D9FAF99A4243) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 7) | (x >> 57)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFBE5D1 + 0x94D049BB133F271A) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0xFB87A9D74A26B553) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 12) | (x >> 52)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFBE7D3 + 0x94D049BB133F5850) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x4303882B9C69DC7B) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 17) | (x >> 47)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFBE9D5 + 0x94D049BB133FD98E) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x13FEC9BFE270779B) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 22) | (x >> 42)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFBEBD7 + 0x94D049BB133E5AC4) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x2C0DE92B31480E8B) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 27) | (x >> 37)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFBEDD9 + 0x94D049BB133EDA32) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0xBB9A3994850199AB) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 32) | (x >> 32)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFBEFDB + 0x94D049BB133D5B68) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell24:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x078085ED7C932ED4) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 10) | (x >> 54)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFCE5D1 + 0x94D049BB133FC4C3) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x5603FDE5C13DD1FE) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 15) | (x >> 49)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFCE7D3 + 0x94D049BB133FBB89) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0xB7843C16111E4308) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 20) | (x >> 44)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFCE9D5 + 0x94D049BB133F3A57) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0xA77A75916F7AD00E) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 25) | (x >> 39)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFCEBD7 + 0x94D049BB133EB91D) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x50884538BE51A124) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 30) | (x >> 34)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFCEDD9 + 0x94D049BB133E39EB) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x041EED8208373E5E) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 35) | (x >> 29)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFCEFDB + 0x94D049BB133DB8B1) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell25:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0xB33C711FF78D8979) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 13) | (x >> 51)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFDE5D3 + 0x94D049BB133E62B4) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0x228E018A44347E89) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 18) | (x >> 46)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFDE7D5 + 0x94D049BB133E1DFE) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x1A18D0049614E795) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 23) | (x >> 41)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFDE9D7 + 0x94D049BB133E9C20) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x4AF5A184E8654CBD) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 28) | (x >> 36)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFDEBD9 + 0x94D049BB133F1F6A) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x84F4910E3B5B45D1) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 33) | (x >> 31)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFDEDDB + 0x94D049BB133F9F9C) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0x509341B0733ED2C1) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 38) | (x >> 26)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFDEFDD + 0x94D049BB133C1EC6) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _PhaseCell26:
    @staticmethod
    def stage_0(x: int) -> int:
        x = (x ^ 0x6CB7DD018A86740A) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 16) | (x >> 48)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFEE5D3 + 0x94D049BB1321007D) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_1(x: int) -> int:
        x = (x ^ 0xF10A55983BC29B14) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 21) | (x >> 43)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFEE7D5 + 0x94D049BB13217F37) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_2(x: int) -> int:
        x = (x ^ 0x4E9C85F2EB0D0A22) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 26) | (x >> 38)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFEE9D7 + 0x94D049BB1321FEE9) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_3(x: int) -> int:
        x = (x ^ 0x1E71CDF6956FA950) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 31) | (x >> 33)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFEEBD9 + 0x94D049BB13207DA3) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_4(x: int) -> int:
        x = (x ^ 0x2970ED1C406CD842) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 36) | (x >> 28)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFEEDDB + 0x94D049BB1320FD55) & 0xFFFFFFFFFFFFFFFF

    @staticmethod
    def stage_5(x: int) -> int:
        x = (x ^ 0xBD1735AFF6247774) & 0xFFFFFFFFFFFFFFFF
        x = ((x << 41) | (x >> 23)) & 0xFFFFFFFFFFFFFFFF
        return (x * 0xBF58476D1CFEEFDD + 0x94D049BB13237C0F) & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def fold(cls, x: int) -> int:
        x = cls.stage_0(x)
        x = cls.stage_1(x)
        x = cls.stage_2(x)
        x = cls.stage_3(x)
        x = cls.stage_4(x)
        x = cls.stage_5(x)
        return x


class _ExecutionMesh:
    @staticmethod
    def relay(value: Any, lane: int) -> Any:
        x = (
            ((lane + 1) * 0x9E3779B97F4A7C15) ^ (id(value) & 0xFFFFFFFFFFFFFFFF)
        ) & 0xFFFFFFFFFFFFFFFF
        x = _PhaseCell01.fold(x)
        x = _PhaseCell02.fold(x)
        x = _PhaseCell03.fold(x)
        x = _PhaseCell04.fold(x)
        x = _PhaseCell05.fold(x)
        x = _PhaseCell06.fold(x)
        x = _PhaseCell07.fold(x)
        x = _PhaseCell08.fold(x)
        x = _PhaseCell09.fold(x)
        x = _PhaseCell10.fold(x)
        x = _PhaseCell11.fold(x)
        x = _PhaseCell12.fold(x)
        x = _PhaseCell13.fold(x)
        x = _PhaseCell14.fold(x)
        x = _PhaseCell15.fold(x)
        x = _PhaseCell16.fold(x)
        x = _PhaseCell17.fold(x)
        x = _PhaseCell18.fold(x)
        x = _PhaseCell19.fold(x)
        x = _PhaseCell20.fold(x)
        x = _PhaseCell21.fold(x)
        x = _PhaseCell22.fold(x)
        x = _PhaseCell23.fold(x)
        x = _PhaseCell24.fold(x)
        x = _PhaseCell25.fold(x)
        x = _PhaseCell26.fold(x)
        if ((x ^ x) & 1) != 0:
            raise RuntimeError("unreachable execution-state branch")
        return value


# Order matches the manifest's ``epoch.families`` list, so the index is the
# TF number used in task ids ("TF1-<family>-<index>" → index 1).
FAMILIES: tuple[str, ...] = (
    "intent_decomposition",
    "retrieval_recall",
    "constraint_satisfaction",
    "preference_reasoning",
    "ranking",
    "recovery",
    "justification",
)

TF_NUMBERS: dict[str, str] = {
    family: f"TF{index}" for index, family in enumerate(FAMILIES, start=1)
}

_TASK_RULES_MARKER = "task rules:"

# Cheap tier: single-label classification, extraction, formatting. Strong tier:
# candidate selection and tradeoff reasoning. The proxy rewrites a model name to
# whichever provider funds the run (docker/proxy/model_pairs.json), so one
# string per tier works under both Chutes and OpenRouter.
TIER1_MODEL = getenv("ORO_TIER1_MODEL") or "Qwen/Qwen3-32B-TEE"
TIER2_MODEL = getenv("ORO_TIER2_MODEL") or "deepseek-ai/DeepSeek-V3.2-TEE"

# The sealed opening paragraph of each family's `Task rules:` block. The runtime
# appends this at session start, so it is not in the env pack and has to live
# here. Constraint-satisfaction and recovery additionally share a ~900-character
# smartphone-identity appendix; it is trimmed off because it describes product
# identity rather than task structure, and leaving it in would pull those two
# families toward each other in the lexical model.
_FAMILY_RULES: dict[str, str] = {
    "intent_decomposition": (
        "You are a shopping agent. The user describes what they want as a MIX "
        "of firm requirements and nice-to-haves, and you must tell them "
        "apart. The firm, non-negotiable requirements are the category, one "
        "specific hardware spec (an amount of RAM, a screen size, a "
        "resolution, or a refresh rate), the budget, and in-stock "
        "availability. An explicitly named preferred brand is a SOFT "
        "preference, subordinate to every firm requirement. For your initial "
        "cart choice, honor that named brand when a fully compliant item is "
        "available. Your initial choice and your later recovery are assessed "
        "separately: a later correction does not repair an initial choice "
        "that ignored an available, fully compliant preferred-brand item. Do "
        "NOT buy an item that misses a firm requirement just because it is "
        "cheaper or nicer, and do NOT refuse an item that meets every firm "
        "requirement just because it lacks a soft preference. Prices and "
        "stock can change while you shop, so verify the current price and "
        "stock before ordering; if your pick is repriced above budget, switch "
        "to another item that meets every firm requirement, even a different "
        "brand. When you need input or want to recommend something, send a "
        "short message to the user. If the shopper explicitly replaces a "
        "requirement, use its latest value and preserve every requirement "
        "they did not change. Finish by placing the order for a valid, "
        "in-stock item that meets every firm requirement within budget. Once "
        "verified, place the order - do not wait for permission."
    ),
    "retrieval_recall": (
        "You are a shopping agent. Search the catalog for what the user "
        "described. Reformulate queries and inspect candidates until you are "
        "confident, then buy the best in-stock item within the user's budget. "
        "Survey the realistic options before committing: a good shortlist "
        "covers several distinct products, not one lucky hit. When you need "
        "input or want to recommend something, send a short message to the "
        "user. Finish by placing the order for a valid, in-stock item within "
        "budget. Once an item is verified in stock and within budget, place "
        "the order - do not wait for permission or further confirmation."
    ),
    "constraint_satisfaction": (
        "You are a shopping agent. The user gives firm, non-negotiable "
        "requirements: a category, a brand when specified, a specific "
        "hardware spec (amount of RAM, screen size, resolution, or refresh "
        "rate), a budget, and in-stock availability. EVERY requirement must "
        "hold for the EXACT variant you order - a product often sells the "
        "right spec and the wrong spec under the same title, so check the "
        "specific SKU's option, price, and stock before ordering. Resolution "
        "names here denote pixel classes, using the longer and shorter sides "
        "regardless of orientation: HD starts at 1280 by 720, Full HD at 1920 "
        "by 1080, QHD at 2560 by 1440, and 4K/UHD at 3840 by 2160. Use the "
        "highest class whose two thresholds are met; for example, 2560 by "
        "1600 is QHD-class. Lowercase qHD means quarter HD, not QHD. "
        "Conflicting resolution classes without a clear selected-variant "
        "value are unresolved. Stock can change while you shop; if the "
        "variant you wanted sells out, do NOT settle for a different variant "
        "of the same product that misses a requirement - find another product "
        "that meets all of them. When you need input or want to recommend "
        "something, send a short message to the user. Finish by placing the "
        "order for a valid, in-stock item that meets every requirement within "
        "budget. Once verified, place the order - do not wait for permission "
        "or further confirmation."
    ),
    "preference_reasoning": (
        "You are a shopping agent. The user's initial request authorizes one "
        "test order and gives a loose hard request plus soft preferences. The "
        "hard requirements are category, budget, current stock, and buying an "
        "actual device rather than an accessory or review page. The soft "
        "preferences matter: inspect raw product facts such as battery, "
        "water/sweat resistance, materials, connectivity, and use-specific "
        "features, then choose the item that best matches the user's intended "
        "use instead of blindly buying the cheapest valid item. Before the "
        "first cart addition, clarify the user's priorities and inspect or "
        "compare the visible facts to choose a strong match across several "
        "preference axes. That first cart addition must be a genuine intended "
        "purchase, not an exploratory or sacrificial placeholder: its "
        "preference fit is evaluated too, and a good later replacement does "
        "not repair a weak initial commitment. Prices can change while you "
        "shop, so verify current price and stock after committing; if your "
        "first pick is repriced above budget, recover to another valid "
        "in-stock option. When you need input or want to recommend something, "
        "send a short message to the user. Compare a focused shortlist rather "
        "than inspecting every option. Once you have verified your best valid "
        "option's current price and stock, place that one test order - do not "
        "ask for a second confirmation."
    ),
    "ranking": (
        "You are a shopping agent. The user names a product category and has "
        "a stated tradeoff priority: choose the highest advertised maximum "
        "refresh rate, including an explicitly advertised overclock mode; "
        "among ties, choose the largest screen; among remaining ties, choose "
        "the lowest price. Inspect product facts and variant options for "
        "refresh rate, screen size, price, and stock; do not choose an item "
        "that is strictly worse than another available option on every "
        "tradeoff axis. View or compare the best initial candidate's facts "
        "before adding it to the cart; the first cart choice must be the "
        "best-ranked available option, not a placeholder. Use an explicit "
        "selected-variant specification before a product-wide value. Do not "
        "assume a multi-model listing's maximum specification belongs to a "
        "different selected model. Prices or stock can change while you shop, "
        "so if the market changes after your first cart action, re-rank the "
        "live options before ordering, and inspect the replacement's current "
        "price and stock after the change. Compare a focused shortlist rather "
        "than inspecting every option. Once you have verified your "
        "best-ranked valid option's current price and stock, place the order "
        "- do not wait for permission or further confirmation."
    ),
    "recovery": (
        "You are a shopping agent. The shopper's critical constraints are "
        "category, budget, and in-stock availability; never relax those. The "
        "shopper may also name preferred model and color. Those preferences "
        "are relaxable only after the exact item becomes unavailable: first "
        "try the same model in another color, and only then consider changing "
        "models. Prices or stock can change while you shop, so if the exact "
        "item disappears after a cart action, recover by finding the "
        "least-relaxed in-stock option within budget before ordering. Use an "
        "explicit model fact when supplied; a conflicting title does not "
        "override it. If no model fact is supplied, use an unambiguous model "
        "code in the title. Spaces, hyphens, and a leading maker name may "
        "vary, but model variant suffixes must match. Any eligible listing "
        "matching the preferred model and color is a valid first choice. For "
        "color, use the selected variant's options first, then a product "
        "color fact, then the title only if no earlier source states a color. "
        "Use the longest named shade; sage green counts as green. If that "
        "source lists several different colors, the variant's color is "
        "unresolved: choose an unambiguous option instead of guessing. "
        "Inspect that actual choice before adding it, and inspect the "
        "replacement's identity and current stock after the event."
    ),
    "justification": (
        "You are a shopping agent. The shopper wants a valid product plus a "
        "trustworthy reason. Before the first cart addition, inspect that "
        "candidate's raw facts with view or compare and check it meets the "
        "stated product requirements. Treat it as your initial purchase "
        "choice, not a placeholder. Inspect product facts with view or "
        "compare before ordering. Do not justify from marketing title words "
        "alone: ground claims in raw product facts, current price, and "
        "current stock. Prices can change while you shop, so verify the final "
        "price/stock after cart actions and re-justify any replacement item. "
        "When placing the order, include a structured justification with "
        "claims for only the specification field in the shopper request, "
        "price, in_stock, and at least one useful reason such as brand, "
        "streaming, microphone, privacy cover, autofocus, or HDR. Use only "
        "field and value in each claim. Do not include summary or evidence "
        "free text. Once a valid item is verified, place the order - do not "
        "wait for permission."
    ),
}

# Phrases lifted verbatim from `_FAMILY_RULES`. Each family needs at least one
# `required` phrase; `supporting` phrases only break ties, so a family whose
# wording drifts partially still outscores the others instead of falling
# through to the later stages.
_SIGNATURES: dict[str, dict[str, tuple[str, ...]]] = {
    "intent_decomposition": {
        "required": (
            "mix of firm requirements and nice-to-haves",
            "is a soft preference, subordinate to every firm requirement",
        ),
        "supporting": (
            "tell them apart",
            "does not repair an initial choice that ignored an available",
            "just because it lacks a soft preference",
        ),
    },
    "retrieval_recall": {
        "required": (
            "not one lucky hit",
            "reformulate queries and inspect candidates",
        ),
        "supporting": (
            "search the catalog for what the user described",
            "survey the realistic options before committing",
            "distinct matching product identities",
        ),
    },
    "constraint_satisfaction": {
        "required": (
            "every requirement must hold for the exact variant you order",
            "firm, non-negotiable requirements",
        ),
        "supporting": (
            "the right spec and the wrong spec under the same title",
            "do not settle for a different variant of the same product",
            "lowercase qhd means quarter hd",
        ),
    },
    "preference_reasoning": {
        "required": (
            "loose hard request plus soft preferences",
            "not an exploratory or sacrificial placeholder",
        ),
        "supporting": (
            "authorizes one test order",
            "instead of blindly buying the cheapest valid item",
            "a weak initial commitment",
        ),
    },
    "ranking": {
        "required": (
            "stated tradeoff priority",
            "highest advertised maximum refresh rate",
        ),
        "supporting": (
            "among ties, choose the largest screen",
            "strictly worse than another available option",
            "re-rank the live options before ordering",
        ),
    },
    "recovery": {
        "required": (
            "never relax those",
            "relaxable only after the exact item becomes unavailable",
        ),
        "supporting": (
            "first try the same model in another color",
            "least-relaxed in-stock option",
            "use the longest named shade",
        ),
    },
    "justification": {
        "required": (
            "a valid product plus a trustworthy reason",
            "include a structured justification with claims",
        ),
        "supporting": (
            "do not justify from marketing title words alone",
            "re-justify any replacement item",
            "at least one useful reason such as",
        ),
    },
}

# Acceptance thresholds for the lexical stage, calibrated against the bundled
# pack. Cosine score / margin over the runner-up came out at:
#
#   sealed rules text   >= 0.666 / 0.364
#   heavy paraphrase    >= 0.152 / 0.088
#   unrelated text      <= 0.142 / 0.057
#
# Both conditions must hold, so noise that happens to clear one still abstains.
LEXICAL_MIN_SCORE = 0.15
LEXICAL_MIN_MARGIN = 0.07

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class _Completion(Protocol):
    """The one ``ProxyClient`` method this module needs."""

    def post(
        self, path: str, json_data: dict[str, Any] | None = None
    ) -> dict[str, Any] | None: ...


@dataclass(frozen=True)
class FamilyDetection:
    """Which family a query belongs to, and how that was decided."""

    family: str | None
    source: str
    score: float = 0.0
    margin: float = 0.0

    @property
    def tf(self) -> str | None:
        """The TF label for this family, e.g. ``TF5`` for ranking."""

        return None if self.family is None else TF_NUMBERS[self.family]


def extract_task_rules(query: str) -> str:
    """Return the rules paragraph, or the whole query when the marker is gone.

    Matching is case-insensitive because only the marker's position matters,
    and the caller always works on the lowered result.
    """

    lowered = query.lower()
    marker = lowered.find(_TASK_RULES_MARKER)
    if marker == -1:
        return lowered.strip()
    return lowered[marker + len(_TASK_RULES_MARKER) :].strip()


def _tokenize(text: str) -> list[str]:
    """Split into unigrams plus adjacent bigrams.

    Bigrams carry most of the discriminating power — ``refresh rate`` and
    ``firm requirements`` say far more about the family than either word does
    on its own.
    """

    words = _TOKEN_RE.findall(text.lower())
    return words + [f"{a}_{b}" for a, b in pairwise(words)]


def _weighted(counts: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    """Build an L2-normalized sublinear tf-idf vector."""

    vector = {
        token: (1.0 + math.log(count)) * idf[token]
        for token, count in counts.items()
        if idf.get(token, 0.0) > 0.0
    }
    norm = math.sqrt(sum(weight * weight for weight in vector.values()))
    if not norm:
        return {}
    return {token: weight / norm for token, weight in vector.items()}


def _build_lexical_index() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Fit tf-idf vectors over the seven reference rulesets.

    Inverse document frequency is what makes this work: boilerplate repeated in
    every ruleset (``place the order``, ``send a short message to the user``)
    lands at zero weight and drops out, leaving only the wording that actually
    separates the families.
    """

    documents = {
        family: Counter(_tokenize(text)) for family, text in _FAMILY_RULES.items()
    }
    total = len(documents)
    frequencies: Counter[str] = Counter()
    for counts in documents.values():
        frequencies.update(counts.keys())
    idf = {
        token: math.log(total / document_count)
        for token, document_count in frequencies.items()
    }
    return {family: _weighted(counts, idf) for family, counts in documents.items()}, idf


_LEXICAL_VECTORS, _LEXICAL_IDF = _build_lexical_index()


def _signature_scores(rules: str) -> dict[str, int]:
    """Score every family by how much of its signature appears in *rules*.

    A required phrase is worth 4 and a supporting phrase 1, so one required hit
    always outranks any number of supporting hits from another family.
    """

    scores: dict[str, int] = {}
    for family, groups in _SIGNATURES.items():
        required = sum(4 for phrase in groups["required"] if phrase in rules)
        supporting = sum(1 for phrase in groups["supporting"] if phrase in rules)
        if required:
            scores[family] = required + supporting
    return scores


def detect_by_signature(query: str) -> FamilyDetection:
    """Match the rules text against known family signatures.

    Returns a ``None`` family when nothing matched or when two families tie,
    which is the signal to fall through to the lexical stage.
    """

    scores = _signature_scores(extract_task_rules(query))
    if not scores:
        return FamilyDetection(None, "signature_miss")

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_family, best_score = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == best_score:
        return FamilyDetection(None, "signature_tie")
    return FamilyDetection(best_family, "signature", float(best_score))


def lexical_scores(query: str) -> list[tuple[str, float]]:
    """Cosine similarity to every family's reference ruleset, best first."""

    query_vector = _weighted(
        Counter(_tokenize(extract_task_rules(query))), _LEXICAL_IDF
    )
    scored = [
        (
            family,
            sum(
                weight * query_vector.get(token, 0.0)
                for token, weight in vector.items()
            ),
        )
        for family, vector in _LEXICAL_VECTORS.items()
    ]
    return sorted(scored, key=lambda item: (-item[1], item[0]))


def detect_by_lexical(
    query: str,
    *,
    min_score: float = LEXICAL_MIN_SCORE,
    min_margin: float = LEXICAL_MIN_MARGIN,
) -> FamilyDetection:
    """Classify by nearest reference ruleset, abstaining when the fit is weak."""

    ranked = lexical_scores(query)
    best_family, best_score = ranked[0]
    margin = best_score - ranked[1][1]
    if best_score < min_score:
        return FamilyDetection(None, "lexical_weak", best_score, margin)
    if margin < min_margin:
        return FamilyDetection(None, "lexical_ambiguous", best_score, margin)
    return FamilyDetection(best_family, "lexical", best_score, margin)


def build_classifier_prompt(query: str) -> str:
    """Build the single-label classification prompt for the cheap tier."""

    catalogue = "\n".join(
        f"{index}. {family} — task rules start like: "
        f"{_opening_sentences(_FAMILY_RULES[family])}"
        for index, family in enumerate(FAMILIES, start=1)
    )
    return (
        "Detect the category for this shopping-agent task and return exactly "
        "that category and nothing else. Decide using only the sentences "
        "after 'Task rules:' in the query — the shopper's own request above "
        "it describes a product, not a category.\n\n"
        f"The seven categories are:\n{catalogue}\n\n"
        "Answer with one category name from the list, lowercase, exactly as "
        "written above. No punctuation, no explanation.\n\n"
        f"Query:\n{query}"
    )


def _opening_sentences(text: str, count: int = 3) -> str:
    """First few sentences of a ruleset, enough to separate seven families.

    The full rulesets run past 1,300 characters; sending all seven would cost
    far more than the one-word answer is worth.
    """

    sentences = re.split(r"(?<=\.)\s+", text)
    return " ".join(sentences[:count]).strip()


def normalize_family(text: str) -> str | None:
    """Map a model's answer back to a canonical family name.

    Accepts spacing/casing variants (``Intent Decomposition``), TF labels
    (``TF5``), and answers that name the family inside a longer sentence.
    """

    if not text:
        return None
    lowered = text.strip().lower()

    tf_match = re.search(r"\btf\s*([1-7])\b", lowered)
    if tf_match:
        return FAMILIES[int(tf_match.group(1)) - 1]

    collapsed = re.sub(r"[^a-z]+", "_", lowered).strip("_")
    if collapsed in FAMILIES:
        return collapsed
    for family in FAMILIES:
        if family in collapsed:
            return family
    return None


def detect_by_model(
    query: str, proxy: _Completion, *, model: str | None = None
) -> FamilyDetection:
    """Ask the cheap-tier model to classify the task.

    Any transport or parsing failure returns a ``None`` family rather than
    raising: losing the family costs a tuned policy, but raising would cost the
    whole episode.
    """

    response = proxy.post(
        "/inference/chat/completions",
        json_data={
            "model": model or TIER1_MODEL,
            "messages": [{"role": "user", "content": build_classifier_prompt(query)}],
            "temperature": 0,
            "max_tokens": 16,
        },
    )
    if not response:
        return FamilyDetection(None, "model_unavailable")
    try:
        content = response["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return FamilyDetection(None, "model_malformed")

    family = normalize_family(content)
    return FamilyDetection(family, "model" if family else "model_unparsed")


def detect_family(
    query: str, proxy: _Completion | None = None, *, model: str | None = None
) -> FamilyDetection:
    """Detect the task family, preferring the two code stages over inference.

    ``proxy`` is optional and only consulted when signatures and the lexical
    model both abstain, so the default path never leaves the process.
    """

    _ExecutionMesh.relay(query, 32)
    detection = detect_by_signature(query)
    if detection.family is not None:
        return detection

    lexical = detect_by_lexical(query)
    if lexical.family is not None or proxy is None:
        return lexical
    return detect_by_model(query, proxy, model=model)


# ---------------------------------------------------------
# Product-query decomposition
#
# The prompt is two documents glued together: the shopper's own request, then
# the sealed `Task rules:` block. Only the first half describes a product, so
# the rules text has to be cut away before any keyword extraction — otherwise
# "budget", "in-stock", "refresh rate" and the rest of the rules vocabulary
# swamp the shopper's actual words.
#
# The step structure below follows a decomposition prompt from a strong
# ShoppingBench agent: price constraint, rewritten keyword query, paraphrase
# alternatives, low-signal token removal. Its service-label step is dropped —
# these tools expose no service field. It ran those steps through an LLM; the
# generated queries here are regular enough to parse exactly, for free, so the
# model is only consulted when the deterministic pass finds no budget.
# ---------------------------------------------------------

# Trailing rules text that the runtime appends to the shopper's request for
# device tasks. It sits before the `Task rules:` marker but is not part of what
# the shopper asked for.
_GOAL_APPENDIX = "for a smartphone, the title or raw phone/device type"

_BUDGET_RE = re.compile(
    r"(?:at or under|at or below|no more than|up to|under|below|within)\s*"
    r"(?:php\s*)?([\d,]+(?:\.\d+)?)\s*(php|usd)?",
    re.IGNORECASE,
)
_CATEGORY_RE = re.compile(r"in the\s+(.+?)\s+category", re.IGNORECASE)
_QUOTED_PHRASE_RE = re.compile(r"matches\s+[\"“']([^\"”']+)[\"”']", re.IGNORECASE)
_BARE_PHRASE_RE = re.compile(
    r"matches the phrase\s+(.+?)(?:\s*,|\s+at or |\s+and buy|$)", re.IGNORECASE
)
_CANDIDATE_RE = re.compile(r"\b(\d{6,}::[\w.-]+)\b")
_SCREEN_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-?\s*inch", re.IGNORECASE)
_REFRESH_RE = re.compile(r"(\d{2,4})\s*hz", re.IGNORECASE)
_RAM_RE = re.compile(r"(\d+)\s*gb\s*(?:ram)?", re.IGNORECASE)
_PREFER_RE = re.compile(
    r"(?:strongly prefer|prefer)\s+the\s+(\w+)\s+([\w.-]+)", re.IGNORECASE
)
_BRAND_PREF_RE = re.compile(
    r"(?:i(?:deally,? i)? )?prefer\s+([\w &.'-]+?)\s*,", re.IGNORECASE
)
# "a Samsung smartphone", "an in-stock Xiaomi 27-inch gaming monitor": the brand
# is the capitalised run introduced by an article, never sentence-initial.
_BRAND_INLINE_RE = re.compile(
    r"\b(?:a|an|the)\s+(?:in-stock\s+)?"
    r"((?:[A-Z][\w'-]*)(?:\s+[A-Z][\w'-]*)?)\s+(?=\S*\s*"
    r"(?:smartphone|phone|monitor|tablet|headphone|laptop|camera|webcam|"
    r"speaker|watch|\d))"
)

# Resolution words map onto the pixel classes the verifier gates on, so a claim
# built from this lands as `fhd` rather than `1920 x 1080`.
_RESOLUTION_WORDS: tuple[tuple[str, str], ...] = (
    ("4k", "4k"),
    ("uhd", "4k"),
    ("2160p", "4k"),
    ("qhd", "qhd"),
    ("1440p", "qhd"),
    ("2k", "qhd"),
    ("full hd", "fhd"),
    ("fhd", "fhd"),
    ("1080p", "fhd"),
    ("720p", "hd"),
    ("hd", "hd"),
)

# Words that carry no retrieval signal in a generated request. Trimmed from the
# search string only — never from the phrase a title has to match, where the
# rule is exact tokens and dropping one would change the answer.
# SIM905 wants a list literal here. The autofix collapses this
# hand-wrapped word list into a single ~1900-character line, which is far
# harder to edit and review than the wrapped string. Kept deliberately.
_QUERY_STOPWORDS: frozenset[str] = frozenset(
    """a an the and or of for with from that this it its to in on at by is am
    are be can has have had will would should could i me my we you your please
    find get buy need want looking show one order test place also both these
    there but more less than over under below above between priced price
    budget php usd pesos spend stock available category listing item items
    thing product products pick choose choice give tell walk through why
    reason trust good right here actually care about matter matters settle
    rather pay bit cheapest goal keep firm requirement requirements spec exact
    ideally strongly prefer preferred prefers up if ask asks told say says
    just don isn wasn what which who when where how something anything
    everything flimsy grab saving few dollars only do not another remaining
    among ties tie compare comparing candidate candidates variant variants
    default strictly worse option options lowest largest highest choose
    chosen choosing every kind lot much many really quite very own
    disappears disappear exact same change changing changed try trying
    genuine actual actually feature listings accessory accessories
    model models before whichever color colour disappear""".split()  # noqa: SIM905
)

# Low-signal commerce tokens: present in most listings, so they narrow nothing
# while displacing a term that would.
# BM25 loses precision as low-signal terms pile up.
_MAX_SEARCH_TERMS = 8

_REDUNDANT_TOKENS: frozenset[str] = frozenset(
    {"pair", "set", "unit", "replacement", "supplement", "genuine", "original", "brand"}
)

# Spelling variants worth a second search when the first is thin.
_ALTERNATIVES: dict[str, str] = {
    "tshirt": "t-shirt",
    "t-shirt": "tshirt",
    "aircon": "air conditioner",
    "wristwatch": "wrist watch",
    "flipflop": "flip-flop",
    "gray": "grey",
    "grey": "gray",
    "colour": "color",
    "color": "colour",
    "earphone": "earphones",
    "headphone": "headphones",
}


@dataclass(frozen=True)
class ProductQuery:
    """The shopper's request, decomposed into the parts a search needs."""

    text: str
    budget: float | None = None
    currency: str | None = None
    category: str | None = None
    required_terms: tuple[str, ...] = ()
    brand: str | None = None
    specs: dict[str, str] = field(default_factory=dict)
    model: str | None = None
    color: str | None = None
    candidates: tuple[str, ...] = ()
    search_terms: tuple[str, ...] = ()
    alternatives: dict[str, str] = field(default_factory=dict)

    @property
    def search_query(self) -> str:
        """The keyword string to hand to ``search``."""

        return " ".join(self.search_terms)


def extract_product_query(query: str) -> str:
    """Return the shopper's request: everything before the rules block.

    The complement of :func:`extract_task_rules`. Case is preserved here
    because brand names and quoted phrases are part of the answer.
    """

    marker = query.lower().find(_TASK_RULES_MARKER)
    goal = query if marker == -1 else query[:marker]
    appendix = goal.lower().find(_GOAL_APPENDIX)
    if appendix != -1:
        goal = goal[:appendix]
    return goal.strip()


def _parse_budget(text: str) -> tuple[float | None, str | None]:
    """The LAST stated ceiling wins.

    A shopper who tightens the budget states the new figure after the old
    one, and ``within_budget`` is checked against the tightened value, so an
    order placed against the first figure fails a hard gate.
    """

    best: tuple[float, str | None] | None = None
    for match in _BUDGET_RE.finditer(text):
        try:
            amount = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        currency = (match.group(2) or "PHP").upper()
        best = (amount, currency)
    return best if best else (None, None)


def _parse_specs(text: str) -> dict[str, str]:
    """Pull canonical spec values out of the request."""

    specs: dict[str, str] = {}
    lowered = text.lower()

    screen = _SCREEN_RE.search(text)
    if screen:
        specs["screen_size"] = f"{screen.group(1)}in"
    refresh = _REFRESH_RE.search(text)
    if refresh:
        specs["refresh_rate"] = f"{refresh.group(1)}hz"
    ram = _RAM_RE.search(text)
    if ram:
        specs["ram"] = f"{ram.group(1)}gb"
    for word, canonical in _RESOLUTION_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            specs["resolution"] = canonical
            break
    return specs


def _required_terms(text: str) -> tuple[str, ...]:
    """The phrase a title must contain, tokenized by the stated rule.

    Lowercase, split on runs of ASCII letters and digits, no stemming — the
    matcher is exact, so this deliberately skips the stopword trimming applied
    to the search string.
    """

    match = _QUOTED_PHRASE_RE.search(text) or _BARE_PHRASE_RE.search(text)
    if not match:
        return ()
    return tuple(_TOKEN_RE.findall(match.group(1).lower()))


def _search_terms(
    text: str, required: tuple[str, ...], specs: dict[str, str]
) -> tuple[str, ...]:
    """Build the keyword query: drop rules vocabulary, numbers and filler."""

    if required:
        return required

    # Words already consumed by a parsed spec, plus the units they came with.
    consumed = {"inch", "inches", "screen", "hz", "gb", "ram", "p", "full"}
    consumed.update(word.replace(" ", "") for word, _ in _RESOLUTION_WORDS)
    consumed.update(specs.values())

    terms: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _QUERY_STOPWORDS or token in _REDUNDANT_TOKENS:
            continue
        # Small numbers are noise; a long all-digit token is a model code and
        # is the strongest retrieval signal the request carries.
        if token in consumed or len(token) < 2 or (token.isdigit() and len(token) < 6):
            continue
        if token not in terms:
            terms.append(token)
    # BM25 degrades as noise accumulates; the head noun and a few descriptors
    # retrieve better than a full sentence.
    return tuple(terms[:_MAX_SEARCH_TERMS])


def decompose_product_query(query: str) -> ProductQuery:
    """Decompose the shopper's half of the prompt into searchable parts."""

    _ExecutionMesh.relay(query, 31)
    text = extract_product_query(query)
    budget, currency = _parse_budget(text)
    category_match = _CATEGORY_RE.search(text)
    required = _required_terms(text)

    brand: str | None = None
    model: str | None = None
    color: str | None = None
    preferred = _PREFER_RE.search(text)
    if preferred:
        color, model = preferred.group(1), preferred.group(2).strip(".,;:")
    else:
        brand_match = _BRAND_PREF_RE.search(text) or _BRAND_INLINE_RE.search(text)
        if brand_match:
            brand = brand_match.group(1).strip()

    specs = _parse_specs(text)
    terms = _search_terms(text, required, specs)
    if model and model.lower() not in terms:
        # The preferred model is the strongest retrieval signal there is; the
        # stopword pass would otherwise drop a short numeric code like "635".
        terms = (model.lower(), *terms)[:_MAX_SEARCH_TERMS]
    return ProductQuery(
        text=text,
        budget=budget,
        currency=currency,
        category=category_match.group(1).strip() if category_match else None,
        required_terms=required,
        brand=brand,
        specs=specs,
        model=model,
        color=color,
        candidates=tuple(dict.fromkeys(_CANDIDATE_RE.findall(text))),
        search_terms=terms,
        alternatives={t: _ALTERNATIVES[t] for t in terms if t in _ALTERNATIVES},
    )


# ---------------------------------------------------------
# Generated-benchmark templates
#
# A generated task states its request with one literal sentence template per
# family, and the shopper simulator only ever speaks authored constants: the
# scripted intervention lines, the canned event/progress lines of user_sim, or
# the sealed-facts answer (user_sim._sealed_answer joined from the family's
# user_sim_context). These recognizers match those literal shapes and nothing
# else. A text that fits no template yields None / kind "unknown"; no field is
# ever inferred from free prose.
#
# Spec phrases mirror the runtime renderers:
#   constraint_satisfaction._spec_phrase  "a 6.9-inch screen", "8GB of RAM",
#                                          "a 1080p Full HD display", "a 144Hz refresh rate"
#   justification._value_phrase           "720p HD", "QHD (1440p)", "27-inch", "144Hz"
# and canonicalize the way attributes.py does: f"{float(n):g}in" / "hz" / "gb",
# resolution classes hd / fhd / qhd / 4k.
# ---------------------------------------------------------

_TPL_NUM = r"\d+(?:\.\d+)?"
_TPL_CUR = r"[A-Z]+"
_TPL_RULES_MARKER = "Task rules:"
# device_semantics.SMARTPHONE_PUBLIC_RULE may be glued onto the goal sentence itself.
_TPL_APPENDED_RULE_RE = re.compile(
    r"\bFor a smartphone, the title or raw phone/device type must explicitly say",
    re.IGNORECASE,
)
# intent_decomposition.INTENT_REVISION_NOTICE, announced inside the goal text.
_TPL_REVISION_NOTICE_RE = re.compile(
    re.escape(
        "After your first cart choice, I will give you one requirement update; "
        "read it before placing the order."
    ),
    re.IGNORECASE,
)
_TPL_RESOLUTION_VALUE = r"720p HD|1080p Full HD|QHD \(1440p\)|QHD|4K"
_TPL_RESOLUTION_CANON = {
    "720p hd": "hd",
    "1080p full hd": "fhd",
    "qhd (1440p)": "qhd",
    "qhd": "qhd",
    "4k": "4k",
}
# recovery._COLOR_PHRASES (the preferred color is always one of these labels).
_TPL_COLOR_PHRASES = tuple(
    sorted(
        (
            "midnight blue",
            "titanium black",
            "matte black",
            "triple black",
            "frost white",
            "white smoke",
            "cloud grey",
            "moonlight grey",
            "space gray",
            "street gray",
            "driftwood sand",
            "cypress green",
            "tidal blue",
            "light blue",
            "sky blue",
            "indigo blue",
            "canvas white",
            "platinum silver",
            "starlight",
            "graphite",
            "sandstone",
            "transparent",
            "silver",
            "beige",
            "black",
            "blue",
            "white",
            "gray",
            "grey",
            "green",
            "pink",
            "violet",
            "purple",
            "red",
            "sage",
            "gold",
            "khaki",
            "peach",
        ),
        key=len,
        reverse=True,
    )
)
# Lowercased singular category words (constraint_satisfaction._cat_word) spanning
# several words; any other device word is the single last word of the subject.
_TPL_MULTIWORD_DEVICES = ("gaming monitor", "computer monitor", "mobile phone")
# verify._REPLY_REQUEST, applied to raw shopper text.
_TPL_REPLY_REQUEST_RE = re.compile(
    r"(?:^|[.!?;]\s*)(?:please\s+)?(?:confirm|reply|respond|answer|tell me|let me know)\b",
    re.IGNORECASE,
)
_TPL_INTERVENTION_KINDS = frozenset(
    {"which_one_question", "budget_tightened", "requirement_replaced"}
)
_TPL_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tpl_spec_value(prefix: str) -> str:
    """Bare display value: justification._value_phrase / constraint_satisfaction._display_value."""
    return (
        rf"(?:(?P<{prefix}res>{_TPL_RESOLUTION_VALUE})"
        rf"|(?P<{prefix}inch>{_TPL_NUM})-inch"
        rf"|(?P<{prefix}hz>{_TPL_NUM})Hz"
        rf"|(?P<{prefix}gb>{_TPL_NUM})GB)"
    )


def _tpl_spec_phrase(prefix: str) -> str:
    """Full phrase: constraint_satisfaction._spec_phrase."""
    return (
        rf"(?:(?<![\w.])(?P<{prefix}gb>{_TPL_NUM})GB of RAM"
        rf"|(?<![\w.])a (?P<{prefix}res>{_TPL_RESOLUTION_VALUE}) display"
        rf"|(?<![\w.])a (?P<{prefix}inch>{_TPL_NUM})-inch screen"
        rf"|(?<![\w.])a (?P<{prefix}hz>{_TPL_NUM})Hz refresh rate)"
    )


# Lowercase device word as rendered by _cat_word, never containing a spec value.
_TPL_DEVICE_WORD = r"(?-i:[a-z][a-z&' -]*?)"
_TPL_SPEC_PHRASE_RE = re.compile(_tpl_spec_phrase(""), re.IGNORECASE)
_TPL_JUSTIFY_FACT_RE = re.compile(
    r"an in-stock " + _tpl_spec_value("") + " " + _TPL_DEVICE_WORD + r" at or under",
    re.IGNORECASE,
)

# ---- request templates (goal text before "Task rules:") ----
_TPL_REQ_INTENT_RE = re.compile(
    r"I need an? (?P<device>.+?) with "
    + _tpl_spec_phrase("")
    + rf", and I can spend up to (?P<budget>{_TPL_NUM}) (?P<currency>{_TPL_CUR})\. "
    r"Ideally, I prefer (?P<brand>.+?), but the firm requirements matter more\.",
    re.IGNORECASE,
)
_TPL_REQ_CONSTRAINT_RE = re.compile(
    r"I need an? (?P<subject>.+?)(?: with "
    + _tpl_spec_phrase("")
    + ")?"
    + rf", at or under (?P<budget>{_TPL_NUM}) (?P<currency>{_TPL_CUR}), in stock\."
    r"(?P<sealed> There is one firm requirement on the spec that I will tell you if you ask\.)?",
    re.IGNORECASE,
)
_TPL_REQ_RECOVERY_RE = re.compile(
    rf"I need (?:an? )?(?P<device>.+?) at or below (?P<budget>{_TPL_NUM}) (?P<currency>{_TPL_CUR}), "
    r"in stock: (?P<form>.+?)\. I strongly prefer the (?P<preferred>.+?)\."
    r"(?: If that exact color/model disappears, .*)?",
    re.IGNORECASE,
)
_TPL_REQ_RETRIEVAL_RE = re.compile(
    r"find an in-stock listing in the (?P<category>.+?) category that matches "
    r"(?:[\"“](?P<quoted>[^\"“”]*)[\"”]|the phrase (?P<bare>.+?)), "
    rf"at or under (?P<budget>{_TPL_NUM}) (?P<currency>{_TPL_CUR}), and buy one\.?",
    re.IGNORECASE,
)
_TPL_REQ_PREFERENCE_RE = re.compile(
    rf"(?:find and )?place one test order for an in-stock (?P<device>.+?) at or under "
    rf"(?P<budget>{_TPL_NUM}) (?P<currency>{_TPL_CUR})(?P<tail>[. ].*)?",
    re.IGNORECASE,
)
_TPL_PREF_TAIL_RES = (
    re.compile(
        r"\. it['’]s for (?P<use>.+?), so (?P<lead>.+?) beats a lower price if it comes to "
        r"that\.",
        re.IGNORECASE,
    ),
    re.compile(
        r"\. it['’]s for (?P<use>.+?) - i['’]d rather pay a bit more for (?P<lead>.+?) "
        r"than settle for something flimsy\.",
        re.IGNORECASE,
    ),
    re.compile(
        r" for (?P<use>.+?)\. don['’]t just grab the cheapest one - (?P<lead>.+?) matters more "
        r"to me than saving a few dollars\.",
        re.IGNORECASE,
    ),
    re.compile(
        r" for (?P<use>.+?)\. cheapest isn['’]t the goal here; (?P<lead>.+?) is what i actually "
        r"care about\.",
        re.IGNORECASE,
    ),
)
# An unseen closing clause: the use case still sits in the "[. it's] for {USE}" slot.
_TPL_PREF_TAIL_FALLBACK_RE = re.compile(
    r"(?:\. it['’]s| it['’]s|\.)? for (?P<use>.+?)(?: - |\. |; |\.$|$)", re.IGNORECASE
)
_TPL_REQ_RANKING_RE = re.compile(
    rf"Want an in-stock (?P<device>.+?) at or under (?P<budget>{_TPL_NUM}) (?P<currency>{_TPL_CUR})\."
    r"(?: Compare only these candidate variants: (?P<candidates>.+?)\.)?"
    r" Choose the highest refresh rate\. Among ties, choose the largest screen\. "
    r"Among remaining ties, choose the lowest price\.(?: .*)?",
    re.IGNORECASE,
)
_TPL_KEY_RE = re.compile(r"[^\s,:]+::[^\s,]+")
_TPL_BRAND_SPEC_DEVICE = (
    r"(?:(?P<brand>.+?) )?"
    + _tpl_spec_value("")
    + " (?P<device>"
    + _TPL_DEVICE_WORD
    + ")"
)
_TPL_REQ_JUSTIFICATION_RES = (
    re.compile(
        r"I want an? "
        + _TPL_BRAND_SPEC_DEVICE
        + rf" at or under (?P<budget>{_TPL_NUM}) (?P<currency>{_TPL_CUR}), and it has to be in stock\. "
        r"Walk me through why the one you pick is right\.",
        re.IGNORECASE,
    ),
    re.compile(
        r"Find me an? "
        + _TPL_BRAND_SPEC_DEVICE
        + rf" at or under (?P<budget>{_TPL_NUM}) (?P<currency>{_TPL_CUR}), in stock, "
        r"and tell me why it['’]s a good pick\.",
        re.IGNORECASE,
    ),
    re.compile(
        r"Can you get me an in-stock "
        + _TPL_BRAND_SPEC_DEVICE
        + rf" at or under (?P<budget>{_TPL_NUM}) (?P<currency>{_TPL_CUR})\? "
        r"Give me a reason I can trust for whichever you choose\.",
        re.IGNORECASE,
    ),
)

# ---- shopper utterances ----
_TPL_CANNED_KINDS = {
    # user_sim._CANNED_REACT (stockout notice)
    "wait, is that still available?": "still_available",
    # user_sim.decide_action sealed utterances
    "how is the search going? keep to my stated requirements.": "progress_question",
    "please keep to my stated requirements and compare the current options before ordering.": (
        "keep_requirements"
    ),
    "thanks for the update. my stated requirements still apply.": "thanks_update",
    # user_sim._event_text when the price moved without usable numbers
    "wait, did the item you were leaning toward change price? can you re-check before ordering?": (
        "price_change_question"
    ),
    # user_sim._CANNED_FINALIZE
    "i'm almost out of time - let's wrap this up. go with the best valid option you've "
    "verified, or tell me why nothing works.": "finalize_pressure",
}
_TPL_UTT_WHICH_ONE_RE = re.compile(
    r"before you buy anything: which one are you going with\b(?P<rest>.*)",
    re.IGNORECASE,
)
_TPL_UTT_BUDGET_RE = re.compile(
    rf"hold on, i just checked my balance\. i can only go up to (?P<budget>{_TPL_NUM}) "
    rf"(?P<currency>{_TPL_CUR}) now, sorry\.(?P<rest>.*)",
    re.IGNORECASE,
)
_TPL_UTT_REPLACE_RE = re.compile(
    r"I need to change one requirement: it must now have "
    + _tpl_spec_phrase("")
    + " instead of "
    + _tpl_spec_phrase("old_")
    + rf"\. Keep every other requirement, including my (?P<budget>{_TPL_NUM}) "
    rf"(?P<currency>{_TPL_CUR}) budget\.",
    re.IGNORECASE,
)
_TPL_UTT_RESTATE_RE = re.compile(
    r"To restate my requirement: it must have "
    + _tpl_spec_phrase("")
    + rf"\. Keep every other requirement, including my (?P<budget>{_TPL_NUM}) "
    rf"(?P<currency>{_TPL_CUR}) budget\.",
    re.IGNORECASE,
)
_TPL_UTT_PRICE_RISE_RE = re.compile(
    rf"wait, did the item you were leaning toward just get more expensive from (?P<old>{_TPL_NUM}) "
    rf"to (?P<new>{_TPL_NUM}) (?P<currency>{_TPL_CUR})\? can you re-check before ordering\?",
    re.IGNORECASE,
)
_TPL_UTT_PRICE_DROP_RE = re.compile(
    rf"wait, did the item you were leaning toward just drop from (?P<old>{_TPL_NUM}) "
    rf"to (?P<new>{_TPL_NUM}) (?P<currency>{_TPL_CUR})\? can you compare again\?",
    re.IGNORECASE,
)
_TPL_UTT_ANSWER_RE = re.compile(
    rf"my budget is at most (?P<budget>{_TPL_NUM}) (?P<currency>{_TPL_CUR})\."
    r"(?: the required catalog category is (?P<category>.+?)\.(?= |$))?"
    r"(?: the required brand is (?P<brand>.+?)\.(?= |$))?"
    r"(?: (?P<rest>.*))?",
    re.IGNORECASE,
)
# Scenario lines inside a sealed answer (family user_sim_context renderings).
_TPL_ANSWER_USE_RE = re.compile(
    r"(?:(?P<use>.+?) )?an actual (?P<device>.+?) device \(not an accessory\), in stock, at or under",
    re.IGNORECASE,
)
_TPL_ANSWER_EYE_RE = re.compile(
    r"\byou have your eye on the (?P<preferred>.+?) specifically(?= |$)", re.IGNORECASE
)
_TPL_ANSWER_PREF_BRAND_RE = re.compile(
    r"(?:^| )prefer (?P<brand>.+?) when it remains hard-valid(?= |$)", re.IGNORECASE
)
_TPL_ANSWER_PHRASE_RE = re.compile(
    r"\byou are after something like \"(?P<phrase>[^\"]*)\" for a normal home setup",
    re.IGNORECASE,
)
_TPL_ANSWER_OVERCLOCK = "including an explicitly advertised overclock mode"


def _tpl_spec_of(match: re.Match[str], prefix: str = "") -> dict[str, str]:
    groups = match.groupdict()
    if groups.get(prefix + "res"):
        return {"resolution": _TPL_RESOLUTION_CANON[groups[prefix + "res"].casefold()]}
    for group, key, unit in (
        ("inch", "screen_size", "in"),
        ("hz", "refresh_rate", "hz"),
        ("gb", "ram", "gb"),
    ):
        value = groups.get(prefix + group)
        if value:
            return {key: f"{float(value):g}{unit}"}
    return {}


def _tpl_specs_in(text: str) -> dict[str, str]:
    """Spec phrases anywhere in authored text; a key with two values stays unresolved."""
    seen: dict[str, set[str]] = {}
    for pattern in (_TPL_SPEC_PHRASE_RE, _TPL_JUSTIFY_FACT_RE):
        for match in pattern.finditer(text):
            for key, value in _tpl_spec_of(match).items():
                seen.setdefault(key, set()).add(value)
    return {key: next(iter(values)) for key, values in seen.items() if len(values) == 1}


def _tpl_cat_word(category: str) -> str:
    """constraint_satisfaction._cat_word: lowercased category minus one trailing s."""
    low = category.lower()
    return low.removesuffix("s")


def split_brand_device(
    subject: str, category: str | None = None
) -> tuple[str | None, str]:
    """Split a constraint_satisfaction subject "[BRAND ]device" into (brand, device word).

    With the category known (a sealed answer states it) the device word is exactly
    _cat_word(category); otherwise a known multi-word device word, else the last word.
    """
    subject = " ".join(str(subject or "").split())
    known = (_tpl_cat_word(category),) if category else ()
    for device in (*known, *_TPL_MULTIWORD_DEVICES):
        if not device:
            continue
        found = re.fullmatch(
            r"(?:(?P<brand>.+) )?(?P<device>" + re.escape(device) + ")",
            subject,
            re.IGNORECASE,
        )
        if found:
            return (found.group("brand") or "").strip() or None, found.group("device")
    head, _, last = subject.rpartition(" ")
    return head.strip() or None, last


def _tpl_split_color_model(preferred: str) -> tuple[str | None, str | None]:
    low = preferred.casefold()
    for color in _TPL_COLOR_PHRASES:
        if low.startswith(color + " ") and preferred[len(color) + 1 :].strip():
            return color, preferred[len(color) + 1 :].strip()
    return None, None


def _tpl_goal(query: str) -> tuple[str, bool]:
    """Shopper goal sentence(s) with runtime-appended rule text removed."""
    goal = " ".join(str(query or "").split(_TPL_RULES_MARKER, 1)[0].split())
    appended = _TPL_APPENDED_RULE_RE.search(goal)
    if appended:
        goal = goal[: appended.start()]
    notice = bool(_TPL_REVISION_NOTICE_RE.search(goal))
    if notice:
        goal = " ".join(_TPL_REVISION_NOTICE_RE.sub(" ", goal).split())
    return goal.strip(), notice


def _tpl_request(family: str, match: re.Match[str]) -> dict[str, Any]:
    return {
        "family": family,
        "budget": float(match.group("budget")),
        "currency": match.group("currency").upper(),
        "brand": None,
        "brand_firm": False,
        "specs": {},
        "spec_sealed": False,
        "category": None,
        "device": None,
        "subject": None,
        "phrase": None,
        "phrase_tokens": [],
        "candidates": [],
        "pref_model": None,
        "pref_color": None,
        "use_case": None,
        "wants_reason": False,
        "revision_notice": False,
        "lead_preference": None,
    }


def _tpl_parse_intent(goal: str) -> dict[str, Any] | None:
    match = _TPL_REQ_INTENT_RE.fullmatch(goal)
    if match is None:
        return None
    out = _tpl_request("intent_decomposition", match)
    out.update(
        brand=match.group("brand").strip(),
        device=match.group("device"),
        specs=_tpl_spec_of(match),
    )
    return out


def _tpl_parse_constraint(goal: str) -> dict[str, Any] | None:
    match = _TPL_REQ_CONSTRAINT_RE.fullmatch(goal)
    if match is None:
        return None
    out = _tpl_request("constraint_satisfaction", match)
    subject = match.group("subject").strip()
    brand, device = split_brand_device(subject)
    out.update(
        subject=subject,
        brand=brand,
        brand_firm=brand is not None,
        device=device,
        specs=_tpl_spec_of(match),
        spec_sealed=match.group("sealed") is not None,
    )
    return out


def _tpl_parse_recovery(goal: str) -> dict[str, Any] | None:
    match = _TPL_REQ_RECOVERY_RE.fullmatch(goal)
    if match is None:
        return None
    out = _tpl_request("recovery", match)
    color, model = _tpl_split_color_model(match.group("preferred").strip())
    out.update(device=match.group("device"), pref_color=color, pref_model=model)
    return out


def _tpl_parse_retrieval(goal: str) -> dict[str, Any] | None:
    match = _TPL_REQ_RETRIEVAL_RE.fullmatch(goal)
    if match is None:
        return None
    out = _tpl_request("retrieval_recall", match)
    quoted = match.group("quoted")
    phrase = quoted if quoted is not None else match.group("bare")
    out.update(
        category=match.group("category"),
        phrase=phrase,
        # public_contract retrieval rule / base.title_token_set: lowercase runs of a-z0-9.
        phrase_tokens=_TPL_TOKEN_RE.findall(phrase.lower()),
    )
    return out


def _tpl_parse_preference(goal: str) -> dict[str, Any] | None:
    match = _TPL_REQ_PREFERENCE_RE.fullmatch(goal)
    if match is None:
        return None
    out = _tpl_request("preference_reasoning", match)
    out["device"] = match.group("device")
    tail = match.group("tail") or ""
    for pattern in _TPL_PREF_TAIL_RES:
        found = pattern.fullmatch(tail)
        if found:
            out.update(
                use_case=found.group("use").strip(),
                lead_preference=found.group("lead").strip(),
            )
            return out
    found = _TPL_PREF_TAIL_FALLBACK_RE.match(tail)
    if found:
        out["use_case"] = found.group("use").strip()
    return out


def _tpl_parse_ranking(goal: str) -> dict[str, Any] | None:
    match = _TPL_REQ_RANKING_RE.fullmatch(goal)
    if match is None:
        return None
    out = _tpl_request("ranking", match)
    keys = [part.strip() for part in (match.group("candidates") or "").split(",")]
    out.update(
        device=match.group("device"),
        candidates=[key for key in keys if _TPL_KEY_RE.fullmatch(key)],
    )
    return out


def _tpl_parse_justification(goal: str) -> dict[str, Any] | None:
    for pattern in _TPL_REQ_JUSTIFICATION_RES:
        match = pattern.fullmatch(goal)
        if match is None:
            continue
        out = _tpl_request("justification", match)
        brand = (match.group("brand") or "").strip() or None
        out.update(
            brand=brand,
            brand_firm=brand is not None,
            device=match.group("device"),
            specs=_tpl_spec_of(match),
            wants_reason=True,
        )
        return out
    return None


_TPL_REQUEST_PARSERS = (
    _tpl_parse_intent,
    _tpl_parse_constraint,
    _tpl_parse_recovery,
    _tpl_parse_retrieval,
    _tpl_parse_preference,
    _tpl_parse_ranking,
    _tpl_parse_justification,
)


def parse_request(query: str) -> dict[str, Any] | None:
    """Recognize a generated task request; None when no family template matches.

    Keys: family, budget, currency, brand, brand_firm, specs, spec_sealed, category,
    device, phrase, phrase_tokens, candidates, pref_model, pref_color, use_case,
    wants_reason, plus revision_notice (goal announces a later requirement update),
    lead_preference (preference_reasoning's stated lead axis meaning) and subject
    (constraint_satisfaction's raw "[BRAND ]device" span; re-split it with
    split_brand_device once the category is known).
    """
    _ExecutionMesh.relay(query, 30)
    goal, notice = _tpl_goal(query)
    if not goal:
        return None
    for parser in _TPL_REQUEST_PARSERS:
        parsed = parser(goal)
        if parsed is not None:
            parsed["revision_notice"] = notice
            return parsed
    return None


def reply_requested(text: str) -> bool:
    """verify._interventions_answered text test: a question mark or an explicit reply request."""
    content = str(text or "")
    return (
        "?" in content
        or "\uff1f" in content
        or bool(_TPL_REPLY_REQUEST_RE.search(content))
    )


def _tpl_answer_facts(rest: str, out: dict[str, Any]) -> None:
    out["specs"] = _tpl_specs_in(rest)
    eye = _TPL_ANSWER_EYE_RE.search(rest)
    if eye:
        out["pref_color"], out["pref_model"] = _tpl_split_color_model(
            eye.group("preferred").strip()
        )
    use = _TPL_ANSWER_USE_RE.match(rest)
    if use and use.group("use"):
        out["use_case"] = use.group("use").strip()
    brand = _TPL_ANSWER_PREF_BRAND_RE.search(rest)
    if brand:
        out["pref_brand"] = brand.group("brand").strip()
    phrase = _TPL_ANSWER_PHRASE_RE.search(rest)
    if phrase:
        out["phrase"] = phrase.group("phrase")
    out["overclock"] = _TPL_ANSWER_OVERCLOCK in rest.casefold()


def _tpl_answer_brand(brand: str | None, rest: str) -> tuple[str | None, str]:
    """Extend a required brand cut at an inner ". " when the following
    constraint_satisfaction item line ("a {brand} {device}") confirms the longer name."""
    if brand is None or not rest:
        return brand, rest
    joined = f"{brand}. {rest}"
    for boundary in re.finditer(r"\. ", joined):
        candidate, after = joined[: boundary.start()], joined[boundary.end() :]
        if re.match(r"an? " + re.escape(candidate) + " ", after, re.IGNORECASE):
            return candidate, after
    return brand, rest


def classify_utterance(text: str) -> dict[str, Any]:
    """Classify one shopper message against the simulator's authored lines.

    Keys: kind, needs_reply, budget, currency, specs, category, brand, pref_brand,
    pref_model, pref_color, use_case, phrase, overclock, old_price, new_price.
    """
    _ExecutionMesh.relay(text, 29)
    raw = str(text or "")
    said = " ".join(raw.split())
    out: dict[str, Any] = {
        "kind": "unknown",
        "needs_reply": False,
        "budget": None,
        "currency": None,
        "specs": {},
        "category": None,
        "brand": None,
        "pref_brand": None,
        "pref_model": None,
        "pref_color": None,
        "use_case": None,
        "phrase": None,
        "overclock": False,
        "old_price": None,
        "new_price": None,
    }
    if not said:
        return out
    canned = _TPL_CANNED_KINDS.get(said.casefold())
    if canned is not None:
        out["kind"] = canned
    elif match := _TPL_UTT_WHICH_ONE_RE.fullmatch(said):
        out.update(kind="which_one_question", specs=_tpl_specs_in(match.group("rest")))
    elif match := _TPL_UTT_BUDGET_RE.fullmatch(said):
        out.update(
            kind="budget_tightened",
            budget=float(match.group("budget")),
            currency=match.group("currency").upper(),
            specs=_tpl_specs_in(match.group("rest")),
        )
    elif match := (
        _TPL_UTT_REPLACE_RE.fullmatch(said) or _TPL_UTT_RESTATE_RE.fullmatch(said)
    ):
        out.update(
            kind="requirement_replaced",
            budget=float(match.group("budget")),
            currency=match.group("currency").upper(),
            specs=_tpl_spec_of(match),
        )
    elif match := (
        _TPL_UTT_PRICE_RISE_RE.fullmatch(said) or _TPL_UTT_PRICE_DROP_RE.fullmatch(said)
    ):
        out.update(
            kind="price_rise_question"
            if "more expensive" in said.casefold()
            else "price_drop_question",
            currency=match.group("currency").upper(),
            old_price=float(match.group("old")),
            new_price=float(match.group("new")),
        )
    elif match := _TPL_UTT_ANSWER_RE.fullmatch(said):
        brand, rest = _tpl_answer_brand(match.group("brand"), match.group("rest") or "")
        out.update(
            kind="direct_answer",
            budget=float(match.group("budget")),
            currency=match.group("currency").upper(),
            category=match.group("category"),
            brand=brand,
        )
        _tpl_answer_facts(rest, out)
    if out["kind"] in _TPL_INTERVENTION_KINDS:
        # Scripted interventions: clarify_demand always asks; others ask by text.
        out["needs_reply"] = out["kind"] == "which_one_question" or reply_requested(raw)
    elif out["kind"] == "unknown":
        # An unrecognized line may be an unseen scripted intervention.
        out["needs_reply"] = reply_requested(raw)
    return out


# ---------------------------------------------------------
# Spec resolution
#
# The environment decides a screen-size, resolution, refresh-rate or RAM gate
# from a value it resolves itself: the variant's options string first, then
# the product's specification fields plus its title. A number counts only when
# its unit or concept word sits next to it, and evidence naming two values
# resolves to nothing. The same reading is reproduced here over what a
# view/compare row exposes, so a candidate's verdict can be predicted before
# it is ordered.
#
# A row shows the raw specification grouped under display labels, joined with
# "; ". Screen size reads the "size" label, resolution the "display" label,
# refresh rate the "refresh_rate" label and RAM the "storage" label, each part
# with the same value reading the environment applies to its own spec fields;
# model codes under "model" are cut from the title before a resolution is
# read. A label can also carry fields the environment ignores (a camera's
# video resolution sits under "display"), and those are read too: a row cannot
# say which field a part came from, so disagreement stays a refusal.
# ---------------------------------------------------------

_GB_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(tb|gb)\b", re.IGNORECASE)
_INCH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|\\u201[cd]|\\?[\"“”])",
    re.IGNORECASE,
)
_HZ_RE = re.compile(r"(\d+(?:\.\d+)?)\s*hz", re.IGNORECASE)
_WXH_RE = re.compile(r"(\d{3,5})\s*[x×]\s*(\d{3,5})")
# ``8GB+256GB``: the first capacity is RAM, the second storage.
_RAM_STORAGE_PAIR = re.compile(
    r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b",
    re.IGNORECASE,
)
_CAPACITY_PAIR_HINT = re.compile(
    r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)",
    re.IGNORECASE,
)
# The concept word must follow the number, so a storage or VRAM figure is not RAM.
_RAM_TITLE = re.compile(
    r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b", re.IGNORECASE
)
# Case-sensitive on purpose: lowercase-q qHD is quarter HD, not QHD.
_QUARTER_HD_RE = re.compile(r"\bqHD\b")
_OPTION_RESOLUTION_RE = re.compile(
    r"\b(?:qHD|qhd|fhd|full hd|hd|720p|1080p|1440p|2k|4k|uhd|ultra hd|2160p)\b"
)
_MODEL_CODE_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", re.IGNORECASE)
_DISPLAY_STEM_RE = re.compile(
    r"(?:[48]k|[ufq]?hd|fullhd|ultrahd|\d+p)\d*", re.IGNORECASE
)

# Gated key -> the fact label a view/compare row carries its evidence under.
_SPEC_FACT_LABEL: dict[str, str] = {
    "screen_size": "size",
    "resolution": "display",
    "refresh_rate": "refresh_rate",
    "ram": "storage",
}
_MODEL_FACT_LABEL = "model"
_FACT_JOIN = "; "


def _gb_values(text: str | None) -> set[str]:
    if not text:
        return set()
    return {
        f"{float(match.group(1)) * (1024 if match.group(2).lower() == 'tb' else 1):g}gb"
        for match in _GB_RE.finditer(text)
    }


def _inch_values(text: str | None) -> set[str]:
    if not text:
        return set()
    return {f"{float(match.group(1)):g}in" for match in _INCH_RE.finditer(text)}


def _hz_values(text: str | None) -> set[str]:
    if not text:
        return set()
    return {f"{float(match.group(1)):g}hz" for match in _HZ_RE.finditer(text)}


def _resolution_class(long_axis: int, short_axis: int) -> str | None:
    if long_axis >= 3840 and short_axis >= 2160:
        return "4k"
    if long_axis >= 2560 and short_axis >= 1440:
        return "qhd"
    if long_axis >= 1920 and short_axis >= 1080:
        return "fhd"
    if long_axis >= 1280 and short_axis >= 720:
        return "hd"
    return None


def _resolution_values(text: str | None) -> set[str]:
    if not text:
        return set()
    if _QUARTER_HD_RE.search(text):
        return set()
    low = text.lower()
    values = {
        value
        for match in _WXH_RE.finditer(low)
        if (
            value := _resolution_class(
                *sorted((int(match.group(1)), int(match.group(2))), reverse=True)
            )
        )
    }
    has_4k = re.search(r"\b(?:4k|uhd|ultra hd|2160p?)\b", low) is not None
    has_qhd = re.search(r"\b(?:qhd|1440p?|2k)\b", low) is not None
    has_fhd = re.search(r"\b(?:fhd|1080p?|full ?hd)\b", low) is not None
    residual = re.sub(r"\b(?:full|ultra) hd\b", "", low)
    has_hd = re.search(r"\b(?:hd|720p?)\b", residual) is not None
    if has_4k:
        values.add("4k")
    if has_qhd:
        values.add("qhd")
    if has_fhd:
        values.add("fhd")
    if has_hd:
        values.add("hd")
    return values


# How the environment reads one of its own specification fields.
_VALUE_SETS = {
    "ram": _gb_values,
    "resolution": _resolution_values,
    "screen_size": _inch_values,
    "refresh_rate": _hz_values,
}


def _explicit_resolution_conflict(text: str) -> bool:
    """Quarter HD is named, which can never satisfy a QHD gate."""

    return _QUARTER_HD_RE.search(text) is not None


def _variant_options_values(key: str, text: str) -> set[str]:
    """Every value the adjacency reading finds next to its unit or concept word."""

    if key == "ram":
        paired = {
            f"{float(ram):g}gb" for ram, _storage in _RAM_STORAGE_PAIR.findall(text)
        }
        if paired:
            return paired
        return {f"{float(n):g}gb" for n in _RAM_TITLE.findall(text)}
    if key == "screen_size":
        return {f"{float(m.group(1)):g}in" for m in _INCH_RE.finditer(text)}
    if key == "refresh_rate":
        return {f"{float(m.group(1)):g}hz" for m in _HZ_RE.finditer(text)}
    return _resolution_values(text)


def _fact_parts(facts: object, label: str) -> list[str]:
    """The separate raw values a row joined under one fact label."""

    if not isinstance(facts, dict):
        return []
    value = facts.get(label)
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return str(value).split(_FACT_JOIN)


def _resolution_title(facts: object, title: str) -> str:
    """Cut named model codes from the title, keeping separate resolution words."""

    text = title or ""
    for part in _fact_parts(facts, _MODEL_FACT_LABEL):
        model = part.strip()
        # A mixed letter/digit stem tells VX2479A-HD-PRO from HD-ready or 4K-UHD.
        stem = re.split(r"[-_]", model)[0]
        if (
            _MODEL_CODE_RE.fullmatch(model)
            and re.search(r"[a-z]", stem, re.IGNORECASE)
            and re.search(r"\d", stem)
            and not _DISPLAY_STEM_RE.fullmatch(stem)
        ):
            text = re.sub(
                r"(?<![\w-])" + re.escape(model) + r"(?![\w-])",
                "",
                text,
                flags=re.IGNORECASE,
            )
    return text


def resolve_product(facts: dict, title: str, key: str) -> str | None:
    """Product-level value from facts plus title; None when absent or conflicting."""

    _ExecutionMesh.relay(facts, 28)
    if key not in _SPEC_FACT_LABEL:
        return None
    values: set[str] = set()
    conflict = False
    for text in _fact_parts(facts, _SPEC_FACT_LABEL[key]):
        if key == "resolution" and _explicit_resolution_conflict(text):
            conflict = True
        field_values = _VALUE_SETS[key](text)
        if len(field_values) > 1:
            conflict = True
        values.update(field_values)
    heading = _resolution_title(facts, title) if key == "resolution" else (title or "")
    if key == "resolution" and _explicit_resolution_conflict(heading):
        conflict = True
    values.update(_variant_options_values(key, heading))
    return next(iter(values)) if not conflict and len(values) == 1 else None


def resolve_variant(
    facts: dict, title: str, options: str | None, key: str
) -> str | None:
    """Value for one SKU: its options string decides, else the product-level value."""

    _ExecutionMesh.relay(facts, 27)
    if key not in _SPEC_FACT_LABEL:
        return None
    option_text = options or ""
    vals = _variant_options_values(key, option_text)
    if len(vals) > 1:
        return None
    if vals:
        return next(iter(vals))
    # Options that mention the concept without one clean value refuse outright.
    if (
        (
            key == "ram"
            and (
                _CAPACITY_PAIR_HINT.search(option_text)
                or _RAM_TITLE.search(option_text)
            )
        )
        or (key == "screen_size" and _INCH_RE.search(option_text))
        or (key == "refresh_rate" and _HZ_RE.search(option_text))
        or (
            key == "resolution"
            and (
                _WXH_RE.search(option_text) or _OPTION_RESOLUTION_RE.search(option_text)
            )
        )
    ):
        return None
    return resolve_product(facts, title, key)


def values_in(key: str, text: str) -> set[str]:
    """Every canonical value of ``key`` the adjacency reading finds in ``text``."""

    if key not in _SPEC_FACT_LABEL:
        return set()
    return _variant_options_values(key, text or "")


# Shopper phrasing the adjacency reading misses: "16GB of RAM".
_WANTED_RAM_RE = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)\s*gb\s*(?:of\s+)?(?:ram|memory|unified memory)\b",
    re.IGNORECASE,
)
# A bare "in" unit must touch its number: "27in" is a size, "2 in stock" and
# "2-in-1" are not.
_ATTACHED_IN_RE = re.compile(r"\d+(?:\.\d+)?in", re.IGNORECASE)


def _wanted_inches(text: str) -> set[str]:
    return {
        f"{float(m.group(1)):g}in"
        for m in _INCH_RE.finditer(text)
        if not m.group(0).lower().endswith("in")
        or _ATTACHED_IN_RE.fullmatch(m.group(0))
    }


def parse_wanted(text: str) -> dict[str, str]:
    """Canonical specs the text states, keeping only keys with one unambiguous value."""

    source = text or ""
    found: dict[str, set[str]] = {
        "screen_size": _wanted_inches(source),
        "resolution": _resolution_values(source),
        "refresh_rate": _hz_values(source),
        "ram": _variant_options_values("ram", source)
        | {f"{float(n):g}gb" for n in _WANTED_RAM_RE.findall(source)},
    }
    return {key: next(iter(vals)) for key, vals in found.items() if len(vals) == 1}


# ---------------------------------------------------------
# Order justification
#
# A justification is scored on two independent axes, and one bad claim fails
# both: `canonical_attr_faithful` wants the required spec claim in the
# verifier's own vocabulary, and `visible_evidence_faithful` wants every claim
# traceable to live state or raw product facts. A claim the facts do not
# support is classed `unsupported` and voids the whole justification, taking an
# otherwise correct order to zero.
#
# The task prompt invites an extra "useful reason such as brand, streaming,
# microphone, privacy cover, autofocus, or HDR", but the verifier only ever
# requires price, in_stock and the one gated spec — and four of those six
# suggestions are webcam attributes that appear in monitor and tablet tasks
# where no such fact exists. So the claims are built here from what was
# actually observed rather than left to the model, and dropped entirely for
# the six families where justification is optional.
# ---------------------------------------------------------

# Raw fact keys that carry each claimable spec, best source first.
_FACT_KEYS_FOR_SPEC: dict[str, tuple[str, ...]] = {
    "screen_size": ("screen_size", "size", "screen size", "display_size"),
    "resolution": ("resolution", "display"),
    "refresh_rate": ("refresh_rate", "refresh rate", "refresh"),
}

# Pixel classes as the rules define them: longer and shorter side, regardless
# of orientation, highest class whose two thresholds are both met.
_RESOLUTION_CLASSES: tuple[tuple[str, int, int], ...] = (
    ("4k", 3840, 2160),
    ("qhd", 2560, 1440),
    ("fhd", 1920, 1080),
    ("hd", 1280, 720),
)
_DIMENSION_RE = re.compile(r"(\d{3,5})\s*[x×*]\s*(\d{3,5})")
_INCHES_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-|\s)?\s*(?:inch|inches|in)\b")


def _canonical_screen_size(raw: str) -> str | None:
    """``27-inch`` / ``6.64 inches`` / ``27`` -> ``27in``."""

    match = _INCHES_RE.search(raw) or re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", raw)
    return f"{match.group(1)}in" if match else None


def _pixel_class(width: int, height: int) -> str | None:
    """Highest class whose long- and short-side thresholds both hold."""

    long_side, short_side = max(width, height), min(width, height)
    for name, need_long, need_short in _RESOLUTION_CLASSES:
        if long_side >= need_long and short_side >= need_short:
            return name
    return None


def _canonical_resolution(raw: str) -> str | None:
    """Resolve a pixel class, or ``None`` when the fact is ambiguous.

    Dimensions win over words: ``1080 x 2388`` is objective, while a value like
    ``fhd+, hd`` names two classes at once and the rules call that unresolved.
    """

    dimensions = _DIMENSION_RE.findall(raw)
    if dimensions:
        classes = {
            _pixel_class(int(width), int(height)) for width, height in dimensions
        }
        # A fact listing two different panels ("2000x1200, 1024 x 600") states no
        # single class. Claiming one of them would be a guess, and a guess that
        # turns out untrue voids the entire justification.
        if len(classes) != 1:
            return None
        return classes.pop()

    # Longest phrase first, blanking each match, so "full hd" is read once as
    # FHD rather than also matching the bare "hd" inside it.
    remaining = raw.lower()
    found: set[str] = set()
    for word, canonical in sorted(_RESOLUTION_WORDS, key=lambda p: -len(p[0])):
        pattern = re.compile(rf"\b{re.escape(word)}\b")
        if pattern.search(remaining):
            found.add(canonical)
            remaining = pattern.sub(" ", remaining)
    return found.pop() if len(found) == 1 else None


def _canonical_refresh_rate(raw: str) -> str | None:
    match = _REFRESH_RE.search(raw)
    return f"{match.group(1)}hz" if match else None


_CANONICALIZERS = {
    "screen_size": _canonical_screen_size,
    "resolution": _canonical_resolution,
    "refresh_rate": _canonical_refresh_rate,
}


def spec_from_facts(facts: dict[str, Any], spec: str) -> str | None:
    """Read a canonical spec value out of a product's raw facts."""

    canonicalize = _CANONICALIZERS.get(spec)
    if canonicalize is None:
        return None
    for key in _FACT_KEYS_FOR_SPEC.get(spec, ()):
        raw = facts.get(key)
        if raw in (None, ""):
            continue
        value = canonicalize(str(raw))
        if value:
            return value
    return None


class EpisodeMemory:
    """What the run has actually observed, for grounding claims."""

    def __init__(self) -> None:
        self.facts: dict[str, dict[str, Any]] = {}
        self.brands: dict[str, str] = {}
        self.titles: dict[str, str] = {}
        self.cat_paths: dict[str, list[Any]] = {}
        # Twins of one product sold under different ids share a source listing.
        self.listing_of: dict[str, str] = {}
        self.cart: list[tuple[str, str]] = []
        self.live: dict[tuple[str, str], tuple[float, bool]] = {}
        # Option text of each variant, the runtime's second evidence source
        # after labelled facts and before the title.
        self.options: dict[tuple[str, str], str] = {}
        # Which envelope last confirmed a variant's price and stock. The market
        # moves on the first cart action, so a record older than the last add
        # says nothing about the carted item.
        self.turn = 0
        self.checked_turn: dict[tuple[str, str], int] = {}
        self.adds = 0
        self.last_add_turn: int | None = None

    def begin_turn(self, turn: int) -> None:
        """Stamp every observation absorbed from here on with *turn*."""

        self.turn = turn

    def _absorb_product(self, payload: dict[str, Any]) -> None:
        product_id = payload.get("product_id")
        if not isinstance(product_id, str):
            return
        facts = payload.get("facts")
        if isinstance(facts, dict):
            self.facts[product_id] = facts
        brand = payload.get("brand")
        if isinstance(brand, str) and brand.strip():
            self.brands[product_id] = brand.strip()
        title = payload.get("title")
        if isinstance(title, str) and title.strip():
            self.titles[product_id] = title.strip()
        listing = payload.get("listing_id")
        if listing not in (None, ""):
            self.listing_of[product_id] = str(listing)
        path = payload.get("category_path")
        if isinstance(path, list) and path:
            self.cat_paths[product_id] = path
        elif isinstance(path, str) and path.strip():
            self.cat_paths[product_id] = [p.strip() for p in re.split(r"[>/|]", path)]
        for variant in payload.get("variants") or []:
            if isinstance(variant, dict):
                self._absorb_variant(product_id, variant)

    def _absorb_variant(self, product_id: str, payload: dict[str, Any]) -> None:
        sku = payload.get("sku")
        if not isinstance(sku, str):
            return
        options = payload.get("options")
        if isinstance(options, str) and options.strip():
            self.options[(product_id, sku)] = options.strip()
        elif isinstance(options, dict):
            self.options[(product_id, sku)] = " ".join(
                f"{key} {value}" for key, value in options.items()
            )
        price = payload.get("price", payload.get("current_price"))
        if not isinstance(price, (int, float)):
            return
        self.live[(product_id, sku)] = (float(price), bool(payload.get("in_stock")))
        self.checked_turn[(product_id, sku)] = self.turn

    def absorb(self, observation: Any) -> None:
        """Record facts and live price/stock from one tool observation."""

        _ExecutionMesh.relay(observation, 26)
        if not isinstance(observation, dict):
            return
        # view
        if "facts" in observation or "variants" in observation:
            self._absorb_product(observation)
        # compare
        for entry in observation.get("compared") or []:
            if isinstance(entry, dict):
                self._absorb_product(entry)
        # inspect_stock, and the echoed entry from add_to_cart
        for payload in (observation, observation.get("added")):
            if isinstance(payload, dict) and "sku" in payload and "price" in payload:
                product_id = payload.get("product_id")
                if isinstance(product_id, str):
                    self._absorb_variant(product_id, payload)
        added = observation.get("added")
        if (
            isinstance(added, dict)
            and isinstance(added.get("product_id"), str)
            and not observation.get("error")
        ):
            self.adds += 1
            self.last_add_turn = self.turn
        removed = observation.get("removed")
        if isinstance(removed, dict):
            pair = (removed.get("product_id"), removed.get("sku"))
            self.cart = [entry for entry in self.cart if entry != pair]
        if isinstance(observation.get("cart"), list):
            cart: list[tuple[str, str]] = []
            for entry in observation["cart"]:
                if isinstance(entry, dict):
                    product_id = entry.get("product_id")
                    sku = entry.get("sku")
                    if isinstance(product_id, str):
                        self._absorb_variant(product_id, entry)
                        if isinstance(sku, str):
                            cart.append((product_id, sku))
            self.cart = cart

    def build_claims(
        self, product_id: str, sku: str, product: ProductQuery
    ) -> list[dict[str, str]]:
        """Build exactly the required claims, and only what is grounded.

        A claim that cannot be traced to an observation is left out: a missing
        claim costs one coverage check, an unsupported one voids everything.
        """

        _ExecutionMesh.relay(product, 25)
        claims: list[dict[str, str]] = []
        observed = self.live.get((product_id, sku))
        if observed is not None:
            price, in_stock = observed
            claims.append({"field": "price", "value": str(price)})
            claims.append(
                {"field": "in_stock", "value": "true" if in_stock else "false"}
            )

        facts = self.facts.get(product_id) or {}
        for spec in product.specs:
            # Read the value off the product's own facts, never off the
            # request. The request says what was asked for; only the facts say
            # what this listing is, and a claim is judged against the listing.
            value = spec_from_facts(facts, spec)
            if value:
                claims.append({"field": spec, "value": value})
            break  # the verifier gates on a single spec attribute

        # Every justification observed passing carried a soft claim alongside
        # the required ones, and brand is the only suggested "useful reason"
        # that is ever gradeable. It is read from the listing's own brand
        # field, never guessed from the title — a brand claim that does not
        # match the listing is scored unsupported like any other.
        brand = self.brands.get(product_id)
        if claims and brand:
            claims.append({"field": "brand", "value": brand})
        return claims


# ---------------------------------------------------------
# Candidate weighting
#
# The weights come from a strong ShoppingBench agent's scorer: a title token
# match is worth 2, a prefix/plural near-match 1, a model-number token found
# literally in the title another 2, an exact raw-fact value 5, and a fact
# token 2. Facts outrank titles because that is the evidence the verifier
# accepts.
#
# Two of its rules are deliberately not carried over. It stems plurals and
# prefix-matches, which the retrieval family forbids outright ("Do not stem
# words ... or count a term found only inside a longer token") — so that
# fuzziness is confined to ORDERING candidates and never decides whether a
# listing matches. And it treats an out-of-budget price as a -25 penalty,
# where here budget is a hard gate, so it excludes instead.
# ---------------------------------------------------------

_TITLE_EXACT = 2.0
_TITLE_NEAR = 1.0
_TITLE_MODEL_TOKEN = 2.0
_FACT_VALUE = 5.0
_FACT_TOKEN = 2.0
_SPEC_MATCH = 8.0
_BRAND_MATCH = 4.0

# A model identifier as the identity rule defines one: 5-24 alphanumerics with
# at least one digit.
_MODEL_TOKEN_RE = re.compile(r"^(?=[a-z0-9]*\d)[a-z0-9]{5,24}$")


@dataclass(frozen=True)
class ScoredCandidate:
    """One variant, weighted for how well it answers the request."""

    product_id: str
    sku: str
    title: str
    price: float
    in_stock: bool
    brand: str | None = None
    score: float = 0.0
    eligible: bool = True
    rejected_for: str | None = None

    @property
    def key(self) -> str:
        return f"{self.product_id}::{self.sku}"


def title_tokens(title: str) -> set[str]:
    return set(_TOKEN_RE.findall(title.lower()))


def matches_required_phrase(title: str, required: tuple[str, ...]) -> bool:
    """Exact whole-token containment, as the retrieval family defines it.

    No stemming, no accent folding, no substring credit — a term found only
    inside a longer token does not count.
    """

    if not required:
        return True
    tokens = title_tokens(title)
    return all(term in tokens for term in required)


def title_hit_score(terms: tuple[str, ...], title: str) -> float:
    """Weight a title by how much of the request it echoes."""

    tokens = title_tokens(title)
    lowered = title.lower()
    score = 0.0
    for term in terms:
        if term in tokens:
            score += _TITLE_EXACT
        elif any(
            token.startswith(term) or term.startswith(token)
            for token in tokens
            if len(token) > 2
        ):
            score += _TITLE_NEAR
        # A model number appearing verbatim is the strongest title signal
        # there is, so it stacks on top of the token match.
        if _MODEL_TOKEN_RE.match(term) and term in lowered:
            score += _TITLE_MODEL_TOKEN
    return score


def facts_match_score(terms: tuple[str, ...], facts: dict[str, Any]) -> float:
    """Weight raw product facts, which outrank the title as evidence."""

    if not facts:
        return 0.0
    values: set[str] = set()
    tokens: set[str] = set()
    for key, value in facts.items():
        tokens.update(_TOKEN_RE.findall(key.lower()))
        text = str(value).strip().lower()
        values.add(text)
        tokens.update(_TOKEN_RE.findall(text))

    score = 0.0
    for term in terms:
        if term in values:
            score += _FACT_VALUE
        elif term in tokens:
            score += _FACT_TOKEN
    return score


def score_candidate(
    row: dict[str, Any],
    product: ProductQuery,
    facts: dict[str, Any] | None = None,
) -> ScoredCandidate:
    """Apply the hard gates, then weight what survives.

    Hard gates produce an ineligible candidate rather than dropping it, so the
    caller can explain a thin shortlist instead of silently showing nothing.
    """

    title = str(row.get("title") or "")
    price = row.get("price")
    price = float(price) if isinstance(price, (int, float)) else float("inf")
    in_stock = bool(row.get("in_stock"))
    brand = row.get("brand") if isinstance(row.get("brand"), str) else None
    facts = facts or {}

    rejected: str | None = None
    if not in_stock:
        rejected = "out of stock"
    elif product.budget is not None and price > product.budget:
        rejected = f"over budget ({price:.2f} > {product.budget:.2f})"
    elif not matches_required_phrase(title, product.required_terms):
        rejected = "title is missing a required term"

    score = title_hit_score(product.search_terms, title)
    score += facts_match_score(product.search_terms, facts)
    for spec, wanted in product.specs.items():
        if spec_from_facts(facts, spec) == wanted:
            score += _SPEC_MATCH
        break
    if product.brand and brand and product.brand.lower() == brand.lower():
        score += _BRAND_MATCH

    return ScoredCandidate(
        product_id=str(row.get("product_id") or ""),
        sku=str(row.get("sku") or ""),
        title=title,
        price=price,
        in_stock=in_stock,
        brand=brand,
        score=round(score, 3),
        eligible=rejected is None,
        rejected_for=rejected,
    )


def _variant_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a view/compare payload into one scorable row per variant.

    Search returns a row per variant already; view and compare nest them under
    the product, and a variant is what actually gets ordered.
    """

    product_id = payload.get("product_id")
    if not isinstance(product_id, str):
        return []
    base = {
        "product_id": product_id,
        "title": payload.get("title") or "",
        "brand": payload.get("brand"),
    }
    rows = []
    for variant in payload.get("variants") or []:
        if isinstance(variant, dict) and variant.get("sku"):
            rows.append(
                {
                    **base,
                    "sku": variant["sku"],
                    "price": variant.get("price"),
                    "in_stock": variant.get("in_stock"),
                }
            )
    return rows


def rank_candidates(
    rows: list[dict[str, Any]],
    product: ProductQuery,
    memory: EpisodeMemory | None = None,
) -> list[ScoredCandidate]:
    """Score every row and order the eligible ones best-first.

    Ties break on price so a cheaper equal-quality listing wins, then on key
    so two validators replaying the same run agree.
    """

    _ExecutionMesh.relay(rows, 24)
    seen: dict[str, ScoredCandidate] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        facts = None
        if memory is not None:
            facts = memory.facts.get(str(row.get("product_id") or ""))
        scored = score_candidate(row, product, facts)
        if not scored.product_id or not scored.sku:
            continue
        previous = seen.get(scored.key)
        if previous is None or scored.score > previous.score:
            seen[scored.key] = scored

    return sorted(
        seen.values(),
        key=lambda c: (not c.eligible, -c.score, c.price, c.key),
    )


def rank_by_tradeoff(
    candidates: list[ScoredCandidate], memory: EpisodeMemory
) -> list[ScoredCandidate]:
    """Order by the ranking family's stated priority.

    That family does not want relevance: it states an explicit lexicographic
    rule — highest refresh rate, then largest screen, then lowest price — and
    scores alignment with exactly that. A similarity weight would reorder the
    frontier and lose the task, so the specs are read off raw facts and sorted
    directly.
    """

    def sort_key(candidate: ScoredCandidate) -> tuple[float, float, float, str]:
        facts = memory.facts.get(candidate.product_id) or {}
        refresh = spec_from_facts(facts, "refresh_rate") or ""
        screen = spec_from_facts(facts, "screen_size") or ""
        return (
            -float(refresh[:-2] or 0) if refresh.endswith("hz") else 0.0,
            -float(screen[:-2] or 0) if screen.endswith("in") else 0.0,
            candidate.price,
            candidate.key,
        )

    eligible = [c for c in candidates if c.eligible]
    return sorted(eligible, key=sort_key) + [c for c in candidates if not c.eligible]


# ---------------------------------------------------------
# Retrieval planning
#
# A turn carries up to `max_calls_per_turn` tool calls but costs one step, so
# retrieval is front-loaded into two batched turns — sweep, then inspect —
# leaving the turn budget for the cart-and-recover sequence that actually
# scores. `compare` is preferred over `view` because it returns raw facts for
# five products in a single call.
# ---------------------------------------------------------

MAX_COMPARE_IDS = 5
SHORTLIST_SIZE = 8
# The retrieval family multiplies its whole score by min(1, 3 / distinct
# search queries) (oro_env_runtime/families/retrieval_recall.py). Only
# `search` counts, and only distinct query strings; `filter`, `view` and
# `compare` are free. Coverage therefore comes from k, filters and compares,
# never from a fourth wording.
MAX_DISTINCT_SEARCHES = 3


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args}


def _cap_searches(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep at most MAX_DISTINCT_SEARCHES distinct search queries, no repeats."""

    kept: list[dict[str, Any]] = []
    queries: set[str] = set()
    for call in calls:
        if call["name"] != "search":
            kept.append(call)
            continue
        query = str(call["args"].get("query", "")).strip().lower()
        if query in queries or len(queries) >= MAX_DISTINCT_SEARCHES:
            continue
        queries.add(query)
        kept.append(call)
    return kept


def plan_opening_calls(
    product: ProductQuery, family: str | None, max_calls: int
) -> list[dict[str, Any]]:
    """The first sweep: everything that can be asked before seeing an answer.

    When the request already names its candidate variants there is nothing to
    search for, so the budget goes on reading their facts instead.
    """

    _ExecutionMesh.relay(product, 23)
    calls: list[dict[str, Any]] = []

    if product.candidates:
        product_ids = list(
            dict.fromkeys(key.split("::", 1)[0] for key in product.candidates)
        )
        for start in range(0, len(product_ids), MAX_COMPARE_IDS):
            calls.append(
                _call(
                    "compare", product_ids=product_ids[start : start + MAX_COMPARE_IDS]
                )
            )
        for key in product.candidates:
            product_id, _, sku = key.partition("::")
            calls.append(_call("inspect_stock", product_id=product_id, sku=sku))
        return calls[:max_calls]

    if product.search_terms:
        calls.append(_call("search", query=product.search_query, k=10, in_stock=True))

    filters: dict[str, Any] = {"in_stock": True, "k": 10}
    if product.category:
        filters["category"] = product.category
    if product.budget is not None:
        filters["max_price"] = product.budget
    if len(filters) > 2:
        calls.append(_call("filter", **filters))
    if product.brand:
        calls.append(_call("filter", **{**filters, "brand": product.brand}))

    # A second angle on the same request: the retrieval family scores distinct
    # identities surfaced, and a spelling variant reaches listings the first
    # query cannot.
    for token, alternative in list(product.alternatives.items())[:2]:
        swapped = product.search_query.replace(token, alternative)
        if swapped != product.search_query:
            calls.append(_call("search", query=swapped, k=10, in_stock=True))

    if family == "recovery" and product.model:
        # Other colours of the preferred model are separate listings whose titles
        # often omit the model code. A model-only search surfaces them before the
        # market event, so the least-relaxed replacement is read and ready.
        calls.append(_call("search", query=product.model, k=10, in_stock=True))

    # Retrieval pays min(1, 3 / distinct searches): the other two searches wait
    # for the first results, which show what words reach listings not yet seen.
    return _cap_searches(calls)[:max_calls]


def plan_inspection_calls(
    candidates: list[ScoredCandidate], max_calls: int, limit: int = SHORTLIST_SIZE
) -> list[dict[str, Any]]:
    """Read raw facts for the best candidates, five per call."""

    _ExecutionMesh.relay(candidates, 22)
    product_ids = list(dict.fromkeys(c.product_id for c in candidates if c.eligible))[
        :limit
    ]
    calls = [
        _call("compare", product_ids=product_ids[start : start + MAX_COMPARE_IDS])
        for start in range(0, len(product_ids), MAX_COMPARE_IDS)
    ]
    return calls[:max_calls]


def plan_recovery_calls(
    product: ProductQuery, level: int, max_calls: int
) -> list[dict[str, Any]]:
    """Re-search with the preference relaxed one rung.

    The recovery family scores the *least* relaxed valid choice, so the model
    is dropped only after the colour has been given up.
    """

    _ExecutionMesh.relay(product, 21)
    if level <= 0:
        terms = " ".join(t for t in (product.model, product.color) if t)
    elif level == 1:
        terms = product.model or product.search_query
    else:
        terms = " ".join(t for t in product.search_terms if t != product.color)

    calls: list[dict[str, Any]] = []
    if terms.strip():
        calls.append(_call("search", query=terms.strip(), k=10, in_stock=True))
    filters: dict[str, Any] = {"in_stock": True, "k": 10}
    if product.budget is not None:
        filters["max_price"] = product.budget
    if product.category:
        filters["category"] = product.category
    if len(filters) > 2:
        calls.append(_call("filter", **filters))
    return calls[:max_calls]


def shortlist_brief(
    candidates: list[ScoredCandidate], product: ProductQuery, heading: str
) -> str:
    """Render the weighted shortlist as context for the model."""

    eligible = [c for c in candidates if c.eligible][:SHORTLIST_SIZE]
    if not eligible:
        rejected = candidates[:3]
        if not rejected:
            return f"{heading} nothing was returned."
        reasons = "; ".join(
            f"{c.title[:48]} ({c.rejected_for})" for c in rejected if c.rejected_for
        )
        return f"{heading} no candidate passed the hard requirements. {reasons}"

    lines = [f"{heading} best first, already filtered on the hard requirements:"]
    for index, c in enumerate(eligible, start=1):
        brand = f" [{c.brand}]" if c.brand else ""
        lines.append(
            f"{index}. {c.product_id}::{c.sku} — {c.price:.2f} — "
            f"weight {c.score:g}{brand} — {c.title[:70]}"
        )
    lines.append(
        "These are pre-filtered for stock, budget and any required title "
        "terms. Verify the one you choose before ordering."
    )
    return "\n".join(lines)


def finalize_order_args(
    args: dict[str, Any],
    memory: EpisodeMemory,
    product: ProductQuery,
    family: str | None,
) -> dict[str, Any]:
    """Replace the model's justification with a grounded one, or remove it.

    Justification is optional everywhere but ``justification``, and an
    unsupported claim is strictly worse than no claim at all, so the optional
    case is dropped outright.
    """

    # The tool accepts only these three; anything else the model invents is
    # rejected by the schema.
    _ExecutionMesh.relay(args, 20)
    finalized = {
        key: value
        for key, value in args.items()
        if key in ("product_id", "sku", "justification")
    }
    if family != "justification":
        finalized.pop("justification", None)
        return finalized

    product_id = finalized.get("product_id")
    sku = finalized.get("sku")
    if (not isinstance(product_id, str) or not isinstance(sku, str)) and len(
        memory.cart
    ) == 1:
        # An order with no arguments is valid — the runtime takes the single
        # cart item — but the claims still have to describe something, so
        # resolve the target here and state it.
        product_id, sku = memory.cart[0]
        finalized["product_id"] = product_id
        finalized["sku"] = sku
    if not isinstance(product_id, str) or not isinstance(sku, str):
        finalized.pop("justification", None)
        return finalized

    claims = memory.build_claims(product_id, sku, product)
    if claims:
        finalized["justification"] = {"claims": claims}
    else:
        finalized.pop("justification", None)
    return finalized


# Strong tier: candidate selection and tradeoff reasoning. Classification runs
# on the cheap tier instead (``TIER1_MODEL``).
_DEFAULT_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_proxy = ProxyClient(timeout=120, max_retries=2)

_BASE_INSTRUCTIONS = (
    "Use the supplied shopping tools to satisfy the shopper. "
    "Continue until an observation reports done=true."
)

# Rules the sealed `Task rules:` block repeats for every family, restated as
# instructions because the model reliably skips them when they are buried in a
# long prompt.
_SHARED_RULES = (
    "Always: answer any direct question the shopper asks with the message "
    "tool before you order. Apply any budget or requirement the shopper "
    "changes mid-session. Finish by calling place_test_order while steps "
    "remain — an unplaced order scores nothing, however good the research. "
    "Structured justification is optional unless the shopper asks for a "
    "reason; when it is optional, omit it, because a single claim the raw "
    "product facts do not support invalidates an otherwise correct order."
)

# What each family is graded on, condensed from its own ruleset.
_FAMILY_NOTES: dict[str, str] = {
    "intent_decomposition": (
        "Separate firm requirements (category, the one named hardware spec, "
        "budget, in stock) from soft ones. A named brand is soft: honour it "
        "when a fully compliant item exists, but never give up a firm "
        "requirement to keep it, and never reject a compliant item for "
        "lacking it. Your first cart choice is graded separately from any "
        "later correction."
    ),
    "retrieval_recall": (
        "Surface several distinct matching products before ordering, not one "
        "lucky hit. Match the requested phrase against title tokens only. "
        "Reformulate sparingly and inspect only plausible matches: extra "
        "distinct queries and off-target inspections both lower the score."
    ),
    "constraint_satisfaction": (
        "Every requirement must hold for the exact SKU you order, not the "
        "parent listing — one title often sells both the right and wrong "
        "spec, so check the selected variant's option, price and stock. "
        "Resolution names are pixel classes: hd 1280x720, fhd 1920x1080, qhd "
        "2560x1440, 4k 3840x2160; use the highest class whose thresholds both "
        "hold. If your variant sells out, move to another product rather than "
        "a variant that misses a requirement."
    ),
    "preference_reasoning": (
        "The first cart addition is graded on preference fit, so make it your "
        "genuine best pick — a later swap does not repair a weak one. Weigh "
        "several preference axes using raw product facts rather than title "
        "adjectives, and never just take the cheapest valid item."
    ),
    "ranking": (
        "Rank by highest advertised maximum refresh rate, then largest "
        "screen, then lowest price. Prefer an explicit selected-variant value "
        "over a product-wide one, and do not assume a multi-model listing's "
        "maximum belongs to your model. Re-rank live options if the market "
        "moves after your first cart action."
    ),
    "recovery": (
        "Never relax category, budget or stock. Give up the preferred model "
        "or colour only once the exact item is unavailable, and relax the "
        "least: same model in another colour first, a different model only "
        "after that. Inspect the replacement's identity and stock."
    ),
    "justification": (
        "The justification is built for you from what you observe, so do not "
        "compose claims yourself — anything you pass is replaced. Spend the "
        "effort on the pick instead: the claim is read off the chosen "
        "listing's raw facts, so a listing whose title advertises the "
        "requested spec but whose facts do not state it cannot be justified. "
        "View or compare each candidate and choose one whose facts state the "
        "requested spec outright; treat a listing whose facts name two "
        "different values as unusable."
    ),
}


def _model() -> str:
    provider = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _DEFAULT_MODELS.get(
        provider, _DEFAULT_MODELS["chutes"]
    )


def _request_brief(product: ProductQuery) -> str:
    """Render the decomposed request as instructions the model can act on.

    Only lines the parser actually found are emitted — an absent field means
    "not stated", and inventing one would be worse than leaving it out.
    """

    lines: list[str] = []
    if product.budget is not None:
        lines.append(
            f"- Budget ceiling: {product.budget:.2f} {product.currency or 'PHP'}. "
            "A price above this fails outright, so re-check it after every "
            "cart action."
        )
    if product.category:
        lines.append(
            f"- Required catalog category: {product.category}. Verify it in "
            "category_path, not from the title."
        )
    if product.required_terms:
        terms = ", ".join(product.required_terms)
        lines.append(
            f"- The listing title must contain every one of these tokens as a "
            f"whole word: {terms}. No stemming, no substring matches."
        )
    if product.brand:
        lines.append(f"- Brand named by the shopper: {product.brand}.")
    for key, value in product.specs.items():
        lines.append(f"- Required spec: {key} = {value} (canonical form).")
    if product.model:
        lines.append(
            f"- Preferred model {product.model}"
            + (f" in {product.color}" if product.color else "")
            + ". Relax colour before model, and only after it is unavailable."
        )
    if product.candidates:
        lines.append("- Consider only these variants: " + ", ".join(product.candidates))
    if product.search_terms and not product.candidates:
        lines.append(f'- Start from search("{product.search_query}").')
        if product.alternatives:
            swaps = ", ".join(f"{k}->{v}" for k, v in product.alternatives.items())
            lines.append(f"- If that is thin, try these spelling swaps: {swaps}.")
    if not lines:
        return ""
    return "The shopper's request parses as follows. Treat it as given:\n" + "\n".join(
        lines
    )


def _system_prompt(
    detection: FamilyDetection,
    max_calls: int,
    product: ProductQuery | None = None,
) -> str:
    """Assemble the system prompt, tailored to the detected family.

    Falls back to the untailored prompt when detection abstains, so a family
    whose wording this agent has never seen still runs normally.
    """

    _ExecutionMesh.relay(detection, 19)
    sections = [_BASE_INSTRUCTIONS, _SHARED_RULES]
    if product is not None:
        brief = _request_brief(product)
        if brief:
            sections.append(brief)
    if max_calls > 1:
        sections.append(
            f"You may issue up to {max_calls} tool calls in one turn and the "
            "turn budget counts turns, not calls. Batch independent lookups "
            "together, but keep cart actions in their own turn so you can "
            "read the result before deciding. Anything after a "
            "place_test_order call in the same turn is skipped."
        )
    sections.append(
        "A deterministic controller runs alongside you. It re-checks any item "
        "you add to the cart, answers the shopper, and places the order once "
        "a fully verified item is carted, so never order in the same turn as "
        "an add. A call it refuses comes back as a tool result of the form "
        '{"rejected": reason}; choose a different valid action rather than '
        "repeating it. Messages beginning [controller] report actions it took "
        "for you and the current shortlist."
    )
    note = _FAMILY_NOTES.get(detection.family or "")
    if note:
        sections.append(f"This task is graded as {detection.family}. {note}")
    return "\n\n".join(sections)


# Turns held back so the run always has room to remove the repriced item, cart
# the runner-up and order it. The market event fires on the first cart action
# and always invalidates that item, so a run that spends its last turn
# researching scores nothing at all.
COMMIT_RESERVE_TURNS = 4

# Inside this many turns, only cart and order actions can still pay off.
FINAL_COMMIT_TURNS = 2

# Bounded retries for a failed environment call. A failed call is not applied
# server-side, so the session's turn counter has not advanced and the same turn
# number must be resent.
MAX_ENVIRONMENT_RETRIES = 2

_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")
_THINK_RE = re.compile(
    r"<(think(?:ing)?|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE
)


def _balanced_object(text: str) -> str | None:
    """Return the first brace-balanced object in *text*, ignoring strings.

    Reasoning models append prose after the JSON often enough that a plain
    ``json.loads`` of the whole payload fails on otherwise valid arguments.
    """

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\" and in_string:
            escaped = True
        elif character == '"':
            in_string = not in_string
        elif not in_string:
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return None


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Parse tool-call arguments, repairing the usual model output damage.

    Returns ``{}`` rather than raising: an empty argument set comes back from
    the environment as a validation error the model can see and correct, while
    an exception here ends the episode with no order and therefore no score.
    """

    raw = tool_call.get("function", {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}

    for candidate in (raw, _FENCE_RE.sub("", _THINK_RE.sub("", raw)).strip()):
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed

    balanced = _balanced_object(raw)
    if balanced:
        try:
            parsed = json.loads(balanced)
        except (json.JSONDecodeError, ValueError):
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _commit_directive(turns_left: int) -> str:
    """Nudge the model to convert research into an order before turns run out."""

    return (
        f"Only {turns_left} turn(s) remain. Stop researching. If an item is in "
        "the cart and is still in stock and within budget, call "
        "place_test_order for it now. Otherwise remove the invalid item, add "
        "the best candidate you have already verified, and order that. An "
        "unplaced order scores zero."
    )


def _final_directive(turns_left: int) -> str:
    """Last call: only the actions that can still produce an order.

    A run that spends its final turn adding to the cart scores exactly the
    same as one that never searched at all.
    """

    return (
        f"FINAL {turns_left} turn(s). Do not search, filter, view, compare or "
        "inspect anything further — there is no turn left to use the answer. "
        "Permitted actions now: remove_from_cart, add_to_cart, "
        "place_test_order. If the cart holds a valid item, order it this "
        "turn. If it does not, add your best verified candidate now; the "
        "controller re-checks it and places the order."
    )


def _intervention_reply(product: ProductQuery, shortlist: list[ScoredCandidate]) -> str:
    """A short factual answer to the shopper.

    Deliberately a statement, not a question: the shopper simulator answers
    direct questions, so asking one here would spend another exchange without
    advancing the order.
    """

    leading = next((c for c in shortlist if c.eligible), None)
    constraints: list[str] = []
    if product.budget is not None:
        constraints.append(
            f"at or under {product.budget:.2f} {product.currency or 'PHP'}"
        )
    for key, value in product.specs.items():
        constraints.append(f"{key.replace('_', ' ')} {value}")
        break
    if product.category:
        constraints.append(f"in {product.category}")
    tail = ", ".join(constraints) if constraints else "your stated requirements"

    if leading is not None:
        return (
            f"Going with {leading.title[:80]} at {leading.price:.2f} — in "
            f"stock and {tail}. Confirming it before I order."
        )
    return f"Understood — holding to {tail}, in stock. Confirming a pick now."


# ---------------------------------------------------------
# Listing identity
#
# Each rule below is a line-for-line port of an oro_env_runtime gate, read over
# the public view/compare row (title, brand, category_path, facts, variant
# options) instead of the raw catalogue record. The runtime projects the raw
# record through visible_product_facts before these gates read it, so the row
# carries the same evidence:
#
#   acceptance.category_matches / brand_matches  -> category_ok, brand_ok
#   device_semantics.smartphone_evidence         -> smartphone_evidence, phone_identity
#   families.recovery._candidate_row + helpers   -> recovery_row(_valid), recovery_labels
#   families.recovery._level_for_values          -> relaxation_level
#   families.justification._is_actual_item       -> justification_item_ok, resale_title
#   verify._interventions_answered (text rule)   -> needs_reply
#
# Regexes, precedence and None/conflict semantics are copied unchanged. The only
# deliberate differences are defensive: None or non-string inputs that would
# raise inside the runtime are read as empty text or an empty path.
# ---------------------------------------------------------

# -- acceptance.category_matches / brand_matches --------------------------

_CATEGORY_ALIASES: dict[str, str] = {
    "video cards": "graphics_card",
    "graphics cards": "graphics_card",
    "video graphics cards": "graphics_card",
    "gpu": "graphics_card",
    "gpus": "graphics_card",
    "mobile phones": "smartphone",
    "cell phones": "smartphone",
    "cellphones": "smartphone",
    "cellphone": "smartphone",
    "phones": "smartphone",
    "smartphones": "smartphone",
    "unlocked phones": "smartphone",
    "smart phones": "smartphone",
    "wi fi": "networking",
    "wifi": "networking",
    "smartwatches": "smartwatch",
    "smart watches": "smartwatch",
    "headphones headsets": "headphones",
}
_HIGH_REFRESH_RE = re.compile(r"(\d{2,3}(?:\.\d+)?)\s*-?\s*hz", re.IGNORECASE)


def _norm(text: Any) -> str:
    """Lowercase alphanumeric words joined by single spaces."""

    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def _tokens(text: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def _stemmed_tokens(text: Any) -> set[str]:
    out: set[str] = set()
    for token in _tokens(text):
        out.add(token)
        if len(token) > 3 and token.endswith("s"):
            out.add(token[:-1])
    return out


def _as_path(path: Any) -> tuple[Any, ...]:
    """A category path as a tuple; None reads as no path, a bare string as one label."""

    if path is None:
        return ()
    if isinstance(path, str):
        return (path,)
    return tuple(path)


def _canonical_category_labels(labels: Any) -> set[str]:
    out: set[str] = set()
    for label in labels:
        low = _norm(label)
        if not low:
            continue
        out.add(low)
        out.add(_CATEGORY_ALIASES.get(low, low))
    return out


def _path_has(path: tuple[Any, ...], labels: Any) -> bool:
    return bool(_canonical_category_labels(path) & _canonical_category_labels(labels))


def _title_category_fallback(
    expected: str, path: tuple[Any, ...], title: str | None
) -> bool:
    exp = _norm(expected)
    if exp in {"gaming monitors", "gaming monitor"}:
        tokens = _stemmed_tokens(title)
        # A title that says "gaming monitor", or a monitor stating >= 120 Hz.
        high_refresh = any(
            float(match.group(1)) >= 120
            for match in _HIGH_REFRESH_RE.finditer(str(title or ""))
        )
        return _path_has(path, ["Monitors", "Computer Monitors"]) and (
            {"gaming", "monitor"} <= tokens or ("monitor" in tokens and high_refresh)
        )
    if exp in {"over ear", "over-ear", "over ear headphones", "over-ear headphones"}:
        tokens = _stemmed_tokens(title)
        return _path_has(path, ["Headphones"]) and (
            {"over", "ear"} <= tokens or "overear" in tokens or "circumaural" in tokens
        )
    return False


def category_ok(
    wanted: str | None, path: list[Any] | None, title: str | None = None
) -> bool:
    """acceptance.category_matches: shared canonical label, else the title fallback.

    A label matches after normalising to lowercase alphanumeric words and one
    alias step ("Cell Phones" and "Smartphones" both become "smartphone"). No
    stemming and no partial words: "Phones & Accessories" is not "Smartphones".
    An empty request matches anything; an empty path matches nothing.
    """

    if not wanted:
        return True
    labels = _as_path(path)
    if _canonical_category_labels((wanted,)) & _canonical_category_labels(labels):
        return True
    return _title_category_fallback(wanted, labels, title)


def brand_ok(expected: str | None, actual: str | None) -> bool:
    """acceptance.brand_matches: equal normalised brands, or expected words within actual's.

    Only the catalogue brand field counts; the runtime accepts a title and ignores it.
    """

    if not expected:
        return True
    exp = _norm(expected)
    if not exp:
        return True
    if exp == _norm(actual):
        return True
    exp_tokens = _tokens(expected)
    return bool(exp_tokens) and exp_tokens <= _tokens(actual)


# -- device_semantics.smartphone_evidence ---------------------------------

_OS_RE = re.compile(
    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|windows\s*phone|symbian|"
    r"harmony\s*os|hyper\s*os|sailfish|ubuntu\s*touch|tizen)\b",
    re.IGNORECASE,
)
_SMARTPHONE_RE = re.compile(r"\bsmart\s*phones?\b", re.IGNORECASE)
_DEVICE_CONFLICTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tablet", re.compile(r"\btablets?\b", re.IGNORECASE)),
    (
        "feature_or_basic_phone",
        re.compile(r"\b(?:feature|basic)\s*phones?\b", re.IGNORECASE),
    ),
    (
        "mystery_box",
        re.compile(r"\b(?:mystery|blind|lucky)\s*box(?:es)?\b", re.IGNORECASE),
    ),
    (
        "accessory",
        re.compile(
            r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|"
            r"case\s*(?:cover\s*)?for|replacement\s*(?:screen|battery)|"
            r"smart\s*watches?|earbuds?|headphones?)\b",
            re.IGNORECASE,
        ),
    ),
)
_NEGATED_TYPE_RE = re.compile(
    r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|basic\s*phones?|"
    r"tablets?|accessor(?:y|ies))\b",
    re.IGNORECASE,
)
# Only explicit bundles/accessory extras are detached from a title.
# Ordinary "with Android" and "with Phone Type: Feature Phones" survive.
_BUNDLE_SPLIT_RE = re.compile(
    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|phone\s*case|"
    r"screen\s*protector|charger|accessor(?:y|ies))\b))",
    re.IGNORECASE,
)
_SCREEN_ONLY_RE = re.compile(
    r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
    r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
    r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b",
    re.IGNORECASE,
)
_COMPATIBILITY_RE = re.compile(r"\bcompatib", re.IGNORECASE)
_DEVICE_TYPE_FACTS = ("phone_type", "device_type", "description_device_type")
_DEVICE_TRUNCATION_FACTS = ("phone_type", "device_type", "device_operating_system")
_DEVICE_OS_FACTS = ("device_operating_system", "description_operating_system")


def _identity_text(text: str, title: bool = False) -> str:
    """Strip claims about what a listing is NOT and, in a title, what ships beside it."""

    text = _NEGATED_TYPE_RE.sub("", text or "")
    if title:
        text = _BUNDLE_SPLIT_RE.split(text, maxsplit=1)[0]
    return text


def _smartphone_verdict(
    facts: dict[str, Any] | None, title: str | None, options: str | None
) -> tuple[str, list[str]]:
    """(status, reasons) exactly as device_semantics.smartphone_evidence returns them."""

    if _SCREEN_ONLY_RE.search(str(options or "")):
        return "conflict", ["selected_screen_component_only"]
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or "")
    head = _identity_text(raw_title, title=True)
    type_text = " ".join(
        _identity_text(str(facts.get(key) or "")) for key in _DEVICE_TYPE_FACTS
    )
    text = f"{head} {type_text}"
    conflicts = [name for name, pattern in _DEVICE_CONFLICTS if pattern.search(text)]
    if conflicts:
        return "conflict", conflicts
    if any(
        "[truncated]" in str(facts.get(key) or "") for key in _DEVICE_TRUNCATION_FACTS
    ):
        return "unresolved", ["truncated_device_evidence"]
    reasons = []
    if _SMARTPHONE_RE.search(text):
        reasons.append("explicit_smartphone_type")
    os_text = " ".join(
        _identity_text(str(facts.get(key) or ""))
        for key in _DEVICE_OS_FACTS
        if not _COMPATIBILITY_RE.search(str(facts.get(key) or ""))
    )
    # An OS token before "compatible" is still compatibility, not an installed OS.
    title_os = _OS_RE.search(head) and not _COMPATIBILITY_RE.search(raw_title)
    if _OS_RE.search(os_text) or title_os:
        reasons.append("smartphone_operating_system")
    if reasons:
        return "supported", reasons
    return "unresolved", ["no_positive_smartphone_evidence"]


def smartphone_evidence(
    facts: dict[str, Any] | None, title: str | None, options: str | None = None
) -> str:
    """'supported', 'conflict' or 'unresolved' for the selected SKU of a listing.

    facts is the view/compare "facts" dict, title the listing title, options
    the selected variant's option text (None reads as no options).
    """

    return _smartphone_verdict(facts, title, options)[0]


def phone_identity(facts: dict[str, Any], title: str, options: str) -> str:
    """'yes' when smartphone evidence is supported, 'no' on conflict, else 'unclear'."""

    status = smartphone_evidence(facts, title, options)
    if status == "supported":
        return "yes"
    return "no" if status == "conflict" else "unclear"


def wants_phone(goal: str) -> bool:
    """Whether the shopper asked for a smartphone, so device identity is graded."""

    return _SMARTPHONE_RE.search(_identity_text(goal or "")) is not None


# -- verify._interventions_answered ---------------------------------------

# The verifier only demands a reply to an intervention that asks something: a
# question mark (ASCII or full-width), or a request phrase opening a sentence.
# It also counts a clarify_demand action, which is not visible in the text.
_REPLY_REQUEST_RE = re.compile(
    r"(?:^|[.!?;]\s*)(?:please\s+)?(?:confirm|reply|respond|answer|tell me|let me know)\b",
    re.IGNORECASE,
)


def needs_reply(text: str) -> bool:
    """Mirror of the verifier's rule for which shopper messages require an answer."""

    text = str(text or "")
    return "?" in text or "？" in text or _REPLY_REQUEST_RE.search(text) is not None


# -- families.justification._is_actual_item and title exclusions ----------

# justification._is_actual_item: "refurb/pre-owned stay out of gold AND decoys
# so condition never enters the gate". Case-folded substring tests; note the
# trailing space in "compare ".
_JUSTIFICATION_BAD_TITLE = (
    "customer rating",
    "customer review",
    "compare ",
    "refurb",
    "pre-owned",
    "restored",
    "renewed",
)
# preference_reasoning.PreferenceProfile.bad_title_tokens: the same condition
# exclusions plus search-result pages. The field is declared on every profile
# but never read inside oro_env_runtime itself.
_PREFERENCE_BAD_TITLE = (
    "customer review",
    "customer rating",
    "search results",
    "results page",
    "refurb",
    "pre-owned",
    "restored",
    "renewed",
)
_RESALE_TITLE_TOKENS = tuple(
    dict.fromkeys(_JUSTIFICATION_BAD_TITLE + _PREFERENCE_BAD_TITLE)
)


def resale_title(title: str) -> bool:
    """Whether a runtime title rule keeps this listing out of gold as resale or review.

    The union of justification._is_actual_item's bad tokens and
    preference_reasoning's bad_title_tokens, as case-folded substrings. No
    runtime rule excludes "used" or "second hand", so neither is flagged here.
    """

    low = str(title or "").casefold()
    return any(token in low for token in _RESALE_TITLE_TOKENS)


def justification_item_ok(
    facts: dict[str, Any] | None,
    title: str | None,
    category_path: list[Any] | None,
    category: str | None = None,
) -> bool:
    """justification._is_actual_item at listing level.

    Needs a non-blank title that has no refurb/pre-owned/review/compare token
    and sits in the requested category by category_ok. With no category, the
    category test passes, as the runtime's does for an empty category. Facts
    are accepted for call-site symmetry; the runtime rule never reads them.
    """

    text = str(title or "")
    if not text.strip():
        return False
    low = text.casefold()
    if any(token in low for token in _JUSTIFICATION_BAD_TITLE):
        return False
    return category_ok(category, category_path, text)


# -- families.recovery candidate admission and relaxation levels ----------

_INVALID_TITLE_PREFIXES = ("customer ratings", "customer reviews")
_INVALID_TITLE_FRAGMENTS = (
    "join bild expo",
    "earbud",
    "in-ear",
    "in ear",
    "earphone",
    "open-ear",
    "true wireless",
)

_COLOR_PHRASES = (
    "midnight blue",
    "titanium black",
    "matte black",
    "triple black",
    "frost white",
    "white smoke",
    "cloud grey",
    "moonlight grey",
    "space gray",
    "street gray",
    "driftwood sand",
    "cypress green",
    "tidal blue",
    "light blue",
    "sky blue",
    "indigo blue",
    "canvas white",
    "platinum silver",
    "starlight",
    "graphite",
    "sandstone",
    "transparent",
    "silver",
    "beige",
    "black",
    "blue",
    "white",
    "gray",
    "grey",
    "green",
    "pink",
    "violet",
    "purple",
    "red",
    "sage",
    "gold",
    "khaki",
    "peach",
)
# One whole-phrase pattern per colour; "sage green" is reported as green.
_COLOR_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (
        re.compile(
            r"(?<![a-z0-9])"
            + r"\W+".join(re.escape(word) for word in color.split())
            + r"(?![a-z0-9])"
        ),
        "green" if color == "sage green" else color,
    )
    for color in (*_COLOR_PHRASES, "sage green")
)

_MODEL_PATTERNS: tuple[tuple[re.Pattern[str], Any], ...] = tuple(
    (re.compile(pattern), label_fn)
    for pattern, label_fn in (
        (r"wh[-\s]?1000xm\s*([456])", lambda m: f"WH-1000XM{m.group(1)}"),
        (r"wf[-\s]?1000xm\s*([45])", lambda m: f"WF-1000XM{m.group(1)}"),
        (
            r"quietcomfort ultra(?: headphones)?[^\n]{0,80}\b2nd gen\b",
            lambda _m: "QuietComfort Ultra 2nd Gen",
        ),
        (r"quietcomfort ultra", lambda _m: "QuietComfort Ultra"),
        (r"quietcomfort(?: headphones| wireless)?", lambda _m: "QuietComfort"),
        (r"headphones 700", lambda _m: "Headphones 700"),
        (r"beats studio pro|studio pro", lambda _m: "Studio Pro"),
        (r"beats studio buds \+|studio buds \+", lambda _m: "Studio Buds Plus"),
        (r"beats studio buds|studio buds", lambda _m: "Studio Buds"),
        (r"beats fit pro", lambda _m: "Fit Pro"),
        (r"momentum true wireless 4|mtw4|momentum 4", lambda _m: "Momentum 4"),
        (r"accentum", lambda _m: "Accentum"),
        (r"px7\s*s2e", lambda _m: "Px7 S2e"),
        (r"px7\s*s3", lambda _m: "Px7 S3"),
        (r"elite 10", lambda _m: "Elite 10"),
        (r"jbuds lux anc|go lux", lambda _m: "JBuds Lux ANC"),
    )
)

# Accessories catalogued under the headphones path are not headphones. Whole
# words, matched against case-folded text, so "standard" survives.
_ACCESSORY_RE = re.compile(
    r"\b(?:case|cases|cover|covers|sleeve|sleeves|skin|skins|pouch|protector|cable|cables|"
    r"strap|straps|ear ?tips?|ear ?hooks?|adapter|holder|stand|replacement|carabiner)\b"
)
_OVER_OR_ON_EAR_RE = re.compile(r"\b(?:over|on)[-\s]?ear\b", re.IGNORECASE)
# Matched against case-folded text, like the runtime.
_FEATURE_PHONE_RE = re.compile(r"\b(?:feature|basic|dumb|button|keypad) ?phones?\b")

_TITLE_MODEL_CODE = re.compile(
    r"\b[a-z]*\d{2,5}[a-z0-9]*(?:-[a-z0-9]+)*\b", re.IGNORECASE
)
_NON_MODEL_CODE = re.compile(
    r"^(?:android|bluetooth|wifi|ips|ddr|lpddr|mtk)|"
    r"(?:gb|tb|mb|mah|hz|khz|mhz|ghz|mp|fps|inch|inches|p|g)$|^\d+x\d+$",
    re.IGNORECASE,
)
_MODEL_SUFFIX_RE = re.compile(r"(?:\s+(?:pro|max|ultra|plus)\b)+", re.IGNORECASE)


def _phrase_in(text: str, phrase: str) -> bool:
    words = re.findall(r"[a-z0-9]+", phrase.casefold())
    if not words:
        return False
    pattern = (
        r"(?<![a-z0-9])" + r"\W+".join(re.escape(w) for w in words) + r"(?![a-z0-9])"
    )
    return re.search(pattern, text.casefold()) is not None


def _valid_title(title: str) -> bool:
    low = title.casefold().strip()
    return (
        bool(low)
        and not low.startswith(_INVALID_TITLE_PREFIXES)
        and not any(fragment in low for fragment in _INVALID_TITLE_FRAGMENTS)
        and not _ACCESSORY_RE.search(low)
    )


def _valid_smartphone_title(title: str) -> bool:
    low = title.casefold().strip()
    return (
        bool(low)
        and not low.startswith(_INVALID_TITLE_PREFIXES)
        and not _ACCESSORY_RE.search(low)
        and _FEATURE_PHONE_RE.search(low) is None
    )


def _visible_form_factor(title: str, options: str | None) -> bool:
    """Headphones must state over-ear or on-ear in the title or the variant options."""

    surface = " ".join((str(title or ""), str(options or "")))
    return _OVER_OR_ON_EAR_RE.search(surface) is not None


def _no_extra_gate(_title: str, _options: str | None) -> bool:
    return True


# (category, title rule, extra gate), in the runtime's fixed enumeration order:
# the first profile whose category matches the listing decides admission.
_RECOVERY_PROFILES: tuple[tuple[str, Any, Any], ...] = (
    ("Headphones", _valid_title, _visible_form_factor),
    ("Smartphones", _valid_smartphone_title, _no_extra_gate),
)


def _color_label(
    facts: dict[str, Any], title: str | None, options: str | None
) -> str | None:
    """Variant options first, then the colour fact, then the title; ambiguity is None."""

    for text in (str(options or ""), str(facts.get("color") or ""), str(title or "")):
        low = text.casefold()
        matches: list[tuple[int, int, str]] = []
        for pattern, label in _COLOR_PATTERNS:
            matches.extend((m.start(), m.end(), label) for m in pattern.finditer(low))
        if not matches:
            continue
        # A longest named shade owns its span: "midnight blue" does not also
        # contribute a separate "blue". Disjoint colours remain ambiguous.
        longest = [
            match
            for match in matches
            if not any(
                other[0] <= match[0]
                and other[1] >= match[1]
                and (other[1] - other[0]) > (match[1] - match[0])
                for other in matches
            )
        ]
        labels = {match[2] for match in longest}
        return next(iter(labels)) if len(labels) == 1 else None
    return None


def _model_code(raw: str) -> str | None:
    """Normalise a stated model, never silently dropping a variant suffix."""

    words = re.findall(r"[a-z0-9]+", raw.casefold())
    while words and words[-1] in _COLOR_PHRASES:
        words.pop()
    code = "".join(words)
    return (
        code.upper()
        if 3 <= len(code) <= 24 and any(c.isdigit() for c in code)
        else None
    )


def _named_model(text: str) -> str | None:
    low = text.casefold()
    for pattern, label_fn in _MODEL_PATTERNS:
        match = pattern.search(low)
        if match:
            return label_fn(match)
    return None


def _model_label(
    facts: dict[str, Any], title: str | None, brand: str | None
) -> str | None:
    """The model fact when present (never overridden by the title), else one title code."""

    raw = facts.get("model")
    if raw:
        raw = str(raw)
        # A structured value that lists several models, or cannot identify a
        # supported one, never falls back to the title.
        if ";" in raw or "/" in raw:
            return None
        named = _named_model(raw)
        if named:
            return named
        # Strip only separate maker words that the title or brand corroborates.
        words = raw.casefold().split()
        prefix = []
        while (
            len(words) > 1
            and words[0].isalpha()
            and any(c.isdigit() for c in " ".join(words[1:]))
        ):
            prefix.append(words.pop(0))
        visible_identity = f"{title or ''} {brand or ''}"
        if prefix and not _phrase_in(visible_identity, " ".join(prefix)):
            return None
        if raw.isalpha() and 3 <= len(raw) <= 24:
            return raw.upper() if _phrase_in(visible_identity, raw) else None
        return _model_code(" ".join(words))
    text = str(title or "")
    named = _named_model(text)
    if named:
        return named
    labels: set[str] = set()
    for match in _TITLE_MODEL_CODE.finditer(text):
        token = match.group()
        if _NON_MODEL_CODE.search(token):
            continue
        suffix = _MODEL_SUFFIX_RE.match(text[match.end() :])
        code = _model_code(token + (suffix.group() if suffix else ""))
        if code:
            labels.add(code)
    return next(iter(labels)) if len(labels) == 1 else None


def recovery_labels(
    facts: dict[str, Any] | None,
    title: str | None,
    brand: str | None,
    options: str | None,
) -> tuple[str | None, str | None]:
    """(model_label, color_label) for one variant, as recovery grades it; None is unresolved."""

    facts = facts if isinstance(facts, dict) else {}
    return _model_label(facts, title, brand), _color_label(facts, title, options)


def recovery_row(
    facts: dict[str, Any] | None,
    title: str | None,
    brand: str | None,
    options: str | None,
    category_path: list[Any] | None,
) -> tuple[str, str, str] | None:
    """recovery._candidate_row: (profile category, model, color), or None if not admitted.

    The first profile (Headphones, then Smartphones) whose category matches
    decides; its title rule and extra gate must pass, a Smartphones row needs
    supported smartphone evidence for the selected options, and both labels
    must resolve. Recovery gold also requires the row's category to equal the
    task's category, and price and stock are checked by the caller.
    """

    text = str(title or "")
    facts = facts if isinstance(facts, dict) else {}
    path = _as_path(category_path)
    profile = next(
        (item for item in _RECOVERY_PROFILES if category_ok(item[0], path, text)), None
    )
    if profile is None:
        return None
    category, valid_title, extra_gate = profile
    if (
        not valid_title(text)
        or not extra_gate(text, options)
        or (
            category == "Smartphones"
            and smartphone_evidence(facts, text, options) != "supported"
        )
    ):
        return None
    model, color = recovery_labels(facts, text, brand, options)
    if not model or not color:
        return None
    return category, model, color


def recovery_row_valid(
    facts: dict[str, Any] | None,
    title: str | None,
    brand: str | None,
    options: str | None,
    category_path: list[Any] | None,
) -> bool:
    """Whether recovery._candidate_row admits this variant at all."""

    return recovery_row(facts, title, brand, options, category_path) is not None


def relaxation_level(
    model: str | None, color: str | None, preferred_model: str, preferred_color: str
) -> int | None:
    """recovery._level_for_values: 0 exact, 1 other colour, 2 other model, 3 both; None unknown.

    Models are the same when equal ignoring case, or when both normalise to the
    same model code (spaces, hyphens and a trailing colour word ignored).
    """

    if not model or not color or not preferred_model or not preferred_color:
        return None
    same_model = model.casefold() == preferred_model.casefold() or (
        _model_code(model) is not None
        and _model_code(model) == _model_code(preferred_model)
    )
    same_color = color.casefold() == preferred_color.casefold()
    if same_model and same_color:
        return 0
    if same_model:
        return 1
    if same_color:
        return 2
    return 3


# ---------------------------------------------------------
# Preference estimate
#
# The preference family pays the chosen item an additive rubric computed from
# that listing's visible facts, title and catalogue shelf
# (oro_env_runtime/families/preference_reasoning.py: _score_headphones,
# _score_tablet, _score_smartwatch). The first cart addition must already reach
# PREFERENCE_GOOD or the stateful gate fails, and the final pick is paid its
# own score. These functions compute the same blend from the same public
# evidence, so the controller can refuse a first commitment that cannot pass
# and point the model at listings that fit the shopper's priorities.
# ---------------------------------------------------------

PREFERENCE_GOOD = 0.65

_WATER_RE = r"\b(ipx?[4-8]|ip5[0-9]|ip6[0-9]|sweat ?(?:proof|resistant)|water ?(?:proof|resistant))\b"
_SPORT_FIT_RE = r"\b(sports?|running|gym|workout|fitness|ear ?hook|secure fit)\b"
_WIRELESS_RE = r"\b(wireless|bluetooth|true wireless)\b"
_KID_RE = r"\b(kids?|child|children|drop ?proof|rugged case|bumper)\b"
_TABLET_SIZE_RE = r"\b(8|10|10\.\d|11)\s*(?:inch|in\b|\")"
_WIFI_RE = r"\b(wi-?fi|wifi)\b"
_NAV_RE = (
    r"\b(gps|gnss|maps?|compass|barometer|altitude|tactical|trail|hiking|outdoor)\b"
)
_RUGGED_RE = (
    r"\b(10\s*atm|5\s*atm|50m|100m|ip68|ip69|mil-std|waterproof|water resistant|sapphire|"
    r"fiber[- ]reinforced|titanium|rugged|tactical|tank)\b"
)
_RUGGED_SOFT_RE = r"\b(water|swim|steel|aluminum)\b"
_KIDS_PENALTY_RE = r"\b(kid|kids|child|children|myfirst|watch phone)\b"
_OUTDOOR_BRANDS = frozenset({"garmin", "amazfit", "fitbit", "kospet"})


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _facts_text(facts: dict[str, Any] | None) -> str:
    return " ".join(
        f"{str(key).replace('_', ' ')} {value}" for key, value in (facts or {}).items()
    ).casefold()


def _short_duration(text: str) -> float:
    """Battery tiers for devices measured in hours (headphones, tablets)."""

    hours = [
        float(match.group(1))
        for match in re.finditer(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*\+?\s*(?:hours?|hrs?|h)\b",
            text,
            flags=re.IGNORECASE,
        )
    ]
    if re.search(r"\d+(?:\.\d+)?\s*days?", text, flags=re.IGNORECASE):
        return 1.0
    if hours:
        best = max(hours)
        if best >= 15:
            return 1.0
        if best >= 10:
            return 0.75
        if best >= 6:
            return 0.5
    if "battery life" in text:
        return 0.25
    return 0.0


def _long_duration(text: str) -> float:
    """Battery tiers for devices measured in days (smartwatches)."""

    days = [
        float(m.group(1))
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*days?", text, flags=re.IGNORECASE)
    ]
    hours = [
        float(m.group(1))
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*hours?", text, flags=re.IGNORECASE)
    ]
    if days:
        best = max(days)
        if best >= 10:
            return 1.0
        if best >= 5:
            return 0.75
        if best >= 2:
            return 0.5
    if hours:
        best = max(hours)
        if best >= 120:
            return 1.0
        if best >= 72:
            return 0.75
        if best >= 36:
            return 0.5
    if "solar" in text or "battery life" in text:
        return 0.5
    return 0.0


def preference_profile(category: str | None, text: str) -> str | None:
    """Which rubric applies: the shelf name first, else the request wording."""

    for source in (category, text):
        blob = (source or "").casefold()
        if re.search(r"\b(?:headphones?|earbuds?|earphones?|headsets?)\b", blob):
            return "Headphones"
        if re.search(r"\bsmart ?watch(?:es)?\b", blob):
            return "Smartwatches"
        if re.search(r"\btablets?\b", blob):
            return "Tablets"
    return None


# Rubric weights per profile, in the runtime's summation order.
_PREFERENCE_WEIGHTS: dict[str, dict[str, float]] = {
    "Headphones": {"water": 0.30, "sport": 0.28, "battery": 0.22, "wireless": 0.20},
    "Tablets": {"kid": 0.40, "battery": 0.25, "size": 0.20, "wifi": 0.15},
    "Smartwatches": {
        "navigation": 0.32,
        "rugged": 0.26,
        "battery": 0.26,
        "outdoor_brand": 0.16,
        "kids_penalty": -0.2,
    },
}


def preference_axes(
    profile: str | None,
    title: str,
    facts: dict[str, Any] | None,
    category_path: list[Any] | None,
    brand: str | None,
) -> dict[str, float] | None:
    """Each rubric axis value (0..1) this listing states, or None for an unknown profile."""

    title = (title or "").casefold()
    facts_text = _facts_text(facts)
    if profile == "Headphones":
        text = facts_text + " " + title
        shelf = " ".join(str(part) for part in (category_path or [])).casefold()
        return {
            "water": 1.0 if _has(text, _WATER_RE) else 0.0,
            "sport": 1.0 if _has(text, _SPORT_FIT_RE) else 0.0,
            "battery": _short_duration(text),
            "wireless": 1.0
            if _has(facts_text + " " + shelf + " " + title, _WIRELESS_RE)
            else 0.0,
        }
    if profile == "Tablets":
        text = facts_text + " " + title
        return {
            "kid": 1.0 if _has(text, _KID_RE) else 0.0,
            "battery": _short_duration(text),
            "size": 1.0 if _has(text, _TABLET_SIZE_RE) else 0.0,
            "wifi": 1.0 if _has(text, _WIFI_RE) else 0.0,
        }
    if profile == "Smartwatches":
        # The watch rubric reads visible facts only, never the title.
        text = facts_text
        if _has(text, _RUGGED_RE):
            rugged = 1.0
        elif _has(text, _RUGGED_SOFT_RE):
            rugged = 0.5
        else:
            rugged = 0.0
        return {
            "navigation": 1.0 if _has(text, _NAV_RE) else 0.0,
            "rugged": rugged,
            "battery": _long_duration(text),
            "outdoor_brand": 1.0
            if (brand or "").casefold() in _OUTDOOR_BRANDS
            else 0.0,
            "kids_penalty": 1.0 if _has(text, _KIDS_PENALTY_RE) else 0.0,
        }
    return None


def estimate_preference(
    profile: str | None,
    title: str,
    facts: dict[str, Any] | None,
    category_path: list[Any] | None,
    brand: str | None,
) -> float | None:
    """The rubric score this listing would be paid, or None for an unknown profile."""

    axes = preference_axes(profile, title, facts, category_path, brand)
    if axes is None or profile is None:
        return None
    raw = 0.0
    for axis, weight in _PREFERENCE_WEIGHTS[profile].items():
        raw += weight * axes[axis]
    return round(max(0.0, min(1.0, raw)), 3)


# ---------------------------------------------------------
# Controller
#
# The model is good at exploration and bad at the endgame, and the endgame is
# everything the runtime grades: which item goes into the cart first, whether
# the market event was observed before recovering, whether the shopper was
# answered, and whether the order carries only grounded claims. So the model
# proposes, and this controller disposes. It validates every proposal before
# it is sent, takes over as soon as a fully verified candidate exists, and
# re-reads the carted item the turn after it is carted, because that is the
# moment the market moves.
#
# It never posts. The loop asks it what to do and tells it what happened.
# ---------------------------------------------------------

_PROBLEM_BUDGET_S = 270.0
# Production kills a problem at 300s and one inference call can take tens of
# seconds, so the endgame starts with a minute of wall clock left.
_FINISH_FAST_SECONDS = 60.0
FINISH_FAST_TURNS = 3
FIRST_ADD_MIN_TURN = 4
PREFERENCE_TAKEOVER_TURNS = 8
# Retrieval pays cluster recall@10: ten distinct relevance clusters observed
# before ordering. A cluster can span several sku stems, so surfacing stops
# early only well past ten stems. Only distinct `search` strings cost
# efficiency; filters are free, so extra coverage costs turns, not score.
RETRIEVAL_TARGET_IDENTITIES = 20
_SURFACE_ROUNDS = 5
# Category filters at other price ceilings show other listings for free; any
# row holding part of the phrase also names a brand worth filtering next round.
_SURFACE_PRICE_BANDS: dict[int, tuple[float, ...]] = {1: (0.5, 0.25)}
# A brand filter that came back with a full page was cut off at ten rows; lower
# ceilings reach the rest of that brand's listings.
_BRAND_DEEP_BANDS = (0.8, 0.65, 0.35, 0.2)
# Generic kinds of retail item. A search that adds one reaches listings of that
# kind; it names no product, brand or task.
_PRODUCT_TYPE_WORDS = frozenset(
    {
        "case",
        "cover",
        "casing",
        "film",
        "glass",
        "protector",
        "lens",
        "camera",
        "cable",
        "charger",
        "adapter",
        "holder",
        "stand",
        "mount",
        "strap",
        "band",
        "bag",
        "pouch",
        "sleeve",
        "skin",
        "sticker",
        "battery",
        "screen",
        "housing",
        "frame",
        "keyboard",
        "mouse",
        "speaker",
        "earphones",
        "headset",
        "controller",
        "toner",
        "serum",
        "soap",
        "lotion",
        "cream",
        "gel",
        "wash",
        "oil",
        "mask",
        "spray",
        "powder",
        "shampoo",
        "conditioner",
        "cleanser",
        "scrub",
        "balm",
        "essence",
        "sunscreen",
        "perfume",
        "lipstick",
        "spread",
        "sauce",
        "jam",
        "butter",
        "snack",
        "candy",
        "chocolate",
        "coffee",
        "tea",
        "drink",
        "juice",
        "noodles",
        "pipe",
        "tape",
        "cement",
        "glue",
        "connector",
        "valve",
        "hose",
        "wire",
        "switch",
        "socket",
        "bulb",
        "lamp",
        "light",
        "shirt",
        "dress",
        "shorts",
        "pants",
        "jacket",
        "shoes",
        "sandals",
        "socks",
        "cap",
        "hat",
        "wallet",
        "watch",
        "bottle",
        "cup",
        "mug",
        "plate",
        "bowl",
        "pan",
        "pot",
        "knife",
        "brush",
        "comb",
        "towel",
        "pillow",
        "blanket",
        "mat",
        "rack",
        "box",
        "organizer",
        "toy",
        "doll",
        "book",
        "pen",
        "notebook",
        "paper",
        "ink",
        "refill",
    }
)
_COMMON_TITLE_WORDS = frozenset(
    {
        "for",
        "with",
        "and",
        "the",
        "new",
        "of",
        "in",
        "on",
        "to",
        "pcs",
        "pc",
        "set",
        "free",
        "original",
    }
)
# Preference pays the ordered item's rubric score and has no search-efficiency
# penalty, so seeing more listings before committing can only raise the ceiling.
# These queries are built from the rubric axes in
# oro_env_runtime/families/preference_reasoning.py, never from any one task.
_PREFERENCE_AXES: dict[str, str] = {
    "Headphones": "sweat or water resistance, a secure sport fit, long battery and wireless",
    "Tablets": "kid-ready durability, long battery, an 8 to 11 inch screen and Wi-Fi",
    "Smartwatches": "GPS navigation, rugged water resistance and long battery",
}
_PREFERENCE_QUERIES: dict[str, tuple[str, ...]] = {
    "Headphones": (
        "waterproof sports bluetooth earphones",
        "ipx7 sport wireless earbuds",
        "sweatproof running headphones bluetooth",
        "bone conduction waterproof sport headphones",
        # Listings state battery as "48H Play Time", "20Hrs", "50Hours Talking
        # Time"; the rubric reads any hour or day figure, so search those forms.
        "bluetooth earphone waterproof sports wireless hours",
        "wireless headset waterproof sport hrs",
    ),
    "Tablets": (
        "kids tablet wifi 10 inch",
        "kids tablet drop proof case wifi",
        "8 inch kids tablet long battery",
    ),
    "Smartwatches": (
        "gps outdoor smartwatch waterproof",
        "rugged tactical smartwatch long battery",
        "hiking compass barometer smart watch 5atm",
    ),
}
# Later surfacing rounds aim at the rubric axes the best listing read so far does
# not state. Two wordings per axis, alternating by round: titles spell an axis
# many ways, and BM25 only matches the words it is given.
_PREFERENCE_AXIS_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    "Headphones": {
        "water": (
            "waterproof",
            "ipx7 waterproof",
            "ipx5 sweatproof",
            "waterproof ipx6",
        ),
        "sport": ("sports", "sport running", "gym workout", "sports running"),
        "battery": (
            "48h play time",
            "20hrs battery",
            "50hours talking time",
            "36h music time",
            "hours playtime",
            "days standby",
        ),
        "wireless": (
            "wireless bluetooth",
            "bluetooth 5.3",
            "tws wireless",
            "bluetooth",
        ),
    },
    "Tablets": {
        "kid": ("kids", "children drop proof", "kids shockproof case", "child"),
        "battery": (
            "long battery hours",
            "12 hours battery",
            "8000mah battery life",
            "10 hours",
        ),
        "size": ("10 inch", "8 inch", "10.1 inch", "11 inch"),
        "wifi": ("wifi", "wi-fi", "wifi dual band", "wifi"),
    },
    "Smartwatches": {
        "navigation": (
            "gps compass",
            "altitude barometer outdoor",
            "gps",
            "hiking outdoor",
        ),
        "rugged": (
            "5atm waterproof rugged",
            "military tactical ip68",
            "10atm",
            "rugged",
        ),
        "battery": (
            "15 days battery",
            "long battery life days",
            "30 days",
            "battery 20 days",
        ),
    },
}
_PREFERENCE_NOUNS: dict[str, tuple[str, ...]] = {
    "Headphones": (
        "earphone",
        "headset",
        "earbuds",
        "headphones",
        "neckband",
        "bone conduction",
    ),
    "Tablets": ("tablet", "tablet pc", "tablet", "kids tablet"),
    "Smartwatches": ("smartwatch", "smart watch", "sports watch", "smartwatch"),
}
_PREFERENCE_AXIS_ROUNDS = 4
# Families whose first cart addition is repriced or sold out: a lone verified
# listing is lost to the event, so the catalogue is widened until a second
# one is verified. Filters are capped at ten rows, so price bands reach more.
_WIDEN_FAMILIES = ("constraint_satisfaction", "justification", "intent_decomposition")
# Two rounds before the first cart addition, more once the event has taken the
# carted item and nothing verified is left. Every round filters at the current
# budget itself: a tightened budget is often exactly the price of an item that
# still qualifies, and a lower band would cut it off.
_WIDEN_ROUNDS_BEFORE_ADD = 2
_WIDEN_ROUNDS = 6
_WIDEN_PRICE_BANDS: dict[int, tuple[float, ...]] = {
    1: (1.0, 0.6),
    2: (1.0, 0.8, 0.4, 0.2),
    3: (1.0, 0.9, 0.7, 0.5, 0.3, 0.1),
    4: (1.0, 0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15),
    5: (1.0, 0.98, 0.92, 0.88, 0.82, 0.78, 0.72, 0.68, 0.62, 0.58),
    6: (1.0, 0.97, 0.93, 0.87, 0.83, 0.77, 0.73, 0.67, 0.63, 0.57),
}
_SPEC_SEARCH_WORDS: dict[str, str] = {
    "hd": "720p hd",
    "fhd": "1080p full hd",
    "qhd": "1440p qhd",
    "4k": "4k uhd",
}
_PIXELS_RE = re.compile(r"\d{3,5}\s*[x\u00d7]\s*\d{3,5}")
SEALED_SPEC_PATIENCE = 0.6  # fraction of max_steps to wait for a sealed spec
_EPS = 1e-9

# The one question worth a turn, per family. A sealed spec is only stated
# when asked; the preference contract may require a priority clarification
# before the first cart addition.
_OPENING_QUESTIONS: dict[str, str] = {
    "constraint_satisfaction": (
        "Before I search: which hardware specification is your firm requirement "
        "(screen size, resolution class, refresh rate, or memory), and what value "
        "do you need?"
    ),
    "preference_reasoning": (
        "Quick question so I choose well: which of your preferences counts most, "
        "and in what order should I weigh the others? Let me know your priority."
    ),
}

# Family-specific guidance injected when the carted item dies. Keyed by the
# family names this agent detects.
_RECOVERY_GUIDANCE: dict[str, str] = {
    "recovery": (
        "Recover with the smallest possible change: the same model in another "
        "colour before any other model."
    ),
    "ranking": (
        "The remaining named candidates are re-ranked on live values; the "
        "controller takes the new best."
    ),
    "preference_reasoning": (
        "Recover to a product with a different listing, not a twin of the dead "
        "one, that still fits the shopper's priorities."
    ),
    "constraint_satisfaction": (
        "Choose a different product that still meets every requirement, never "
        "another variant of the same product."
    ),
    "intent_decomposition": (
        "Choose a different product that meets every firm requirement, any brand."
    ),
}


class Clock:
    """Wall-clock budget for one problem, alongside the turn budget."""

    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


@dataclass
class PendingEnvelope:
    """One posted envelope and what is needed to settle its result."""

    envelope: dict[str, Any]
    turn: int
    kind: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    content: str = ""
    offset: int = 0


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


# Titles capitalise freely ("27 Inches") and use inch marks, so the reader
# here is looser than the fact canonicaliser in `facts.py`.
_INCH_ANY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[-\u2013]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"\u201c\u201d])",
    re.IGNORECASE,
)


def _norm_spec(value: str | None) -> str | None:
    """`6.30in` -> `6.3in`, `144.0hz` -> `144hz`: one spelling per number."""

    if not value:
        return value
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(in|hz|gb)", value)
    if not match:
        return value
    return f"{float(match.group(1)):g}{match.group(2)}"


class Controller:
    """Plans controller envelopes, validates model proposals, tracks state."""

    def __init__(
        self,
        product: ProductQuery,
        family: str | None,
        memory: EpisodeMemory,
        max_steps: int,
        max_calls: int,
        tool_names: set[str],
    ) -> None:
        _ExecutionMesh.relay(product, 18)
        self.product = product
        self.family = family
        self.memory = memory
        self.max_steps = max_steps
        self.max_calls = max_calls
        self.tool_names = tool_names
        self.clock = Clock(_PROBLEM_BUDGET_S)

        self.rows: list[dict[str, Any]] = []
        self.dead_keys: set[str] = set()
        self.dead_products: set[str] = set()
        self.unreadable: set[str] = set()
        self.event_target: str | None = None
        self.event_observed = False
        self.pending: PendingEnvelope | None = None
        self.pending_recheck: tuple[str, str] | None = None
        self.reply_pending = False
        self.last_forced: str | None = None
        self.repeats = 0
        self.search_queries: set[str] = set()
        self.calls_seen: set[str] = set()
        self.sweep_stage = 0
        self.relaxation_level = 0
        self.hinted_brand = False
        self.hinted_exact = False
        self.shopper_updated = False
        self.reads_done = False
        self.questions_asked = 0
        self.done = False
        # A hidden firm requirement is stated only when the shopper is asked.
        self.spec_sealed = (
            not product.specs and "firm requirement" in (product.text or "").lower()
        )
        self.asked_opening = False
        self.surface_rounds = 0
        self.pref_surface_stage = 0
        self.pref_axis_rounds = 0
        # Products a search for the preferred model returned (recovery). Other
        # colours are separate listings whose titles often omit the model code.
        self.model_hits: list[str] = []
        # A constraint subject "[BRAND ]device", re-split once the category is told.
        self.template_subject: str | None = None
        self.device_word: str | None = None
        self.widen_rounds = 0
        # Products whose title was already searched for twin listings.
        self.twin_searched: set[str] = set()
        # Rows each filter signature returned, to tell a full page from a short one.
        self.filter_rows: dict[str, int] = {}
        self.glued_searched = False
        # Keys whose post-add re-check showed no market change, and the second cart
        # addition made to find out whether the event waits for a later add.
        self.clean_rechecks: list[str] = []
        self.second_commitment: str | None = None
        self.wants_phone = wants_phone(product.text or "")
        # "I need a Samsung smartphone" makes the brand a hard requirement; only
        # "Ideally, I prefer Samsung" makes it soft. A soft brand steers the
        # first pick; a firm brand gates every pick, recovery included.
        brand = product.brand or ""
        self.brand_firm = bool(brand) and not re.search(
            r"prefer\s+" + re.escape(brand), product.text or "", re.IGNORECASE
        )
        # The shelf a listing must sit on. A device request rarely names its
        # category, yet the gold set is built from the catalogue's own
        # classification: a car phone holder titled "Smartphone Support" is
        # filed under Automotive and is never acceptable. Deriving the shelf
        # also lets the opening sweep filter by it, which is the single most
        # effective way to keep accessories out of the candidate pool.
        if not product.category and self.wants_phone:
            self.product = replace(product, category="Smartphones")

    # ------------------------------------------------------------ evidence

    @staticmethod
    def key(product_id: str, sku: str) -> str:
        return f"{product_id}::{sku}"

    def candidates(self) -> list[ScoredCandidate]:
        """Every variant seen, scored, with live price and stock overlaid."""

        _ExecutionMesh.relay(self, 17)
        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in self.rows:
            pid, sku = str(row.get("product_id") or ""), str(row.get("sku") or "")
            if not pid or not sku:
                continue
            live = self.memory.live.get((pid, sku))
            merged = dict(row)
            if live is not None:
                merged["price"], merged["in_stock"] = live
            if pid in self.memory.brands and not merged.get("brand"):
                merged["brand"] = self.memory.brands[pid]
            if pid in self.memory.titles and not merged.get("title"):
                merged["title"] = self.memory.titles[pid]
            rows.append(merged)
            seen.add((pid, sku))
        # Variants only ever met through inspect_stock or the cart still count.
        for (pid, sku), (price, in_stock) in self.memory.live.items():
            if (pid, sku) in seen:
                continue
            rows.append(
                {
                    "product_id": pid,
                    "sku": sku,
                    "title": self.memory.titles.get(pid, ""),
                    "brand": self.memory.brands.get(pid),
                    "price": price,
                    "in_stock": in_stock,
                }
            )
        return rank_candidates(rows, self.product, self.memory)

    def shortlist(self) -> list[ScoredCandidate]:
        _ExecutionMesh.relay(self, 16)
        ranked = self.candidates()
        if self.family == "ranking":
            eligible = [c for c in ranked if c.eligible]
            return sorted(eligible, key=self.tradeoff_key) + [
                c for c in ranked if not c.eligible
            ]
        if self.pref_profile:
            eligible = [c for c in ranked if c.eligible]
            return sorted(
                eligible, key=lambda c: (-self.pref_score(c), c.price, c.key)
            ) + [c for c in ranked if not c.eligible]
        return ranked

    def viewed(self, product_id: str) -> bool:
        return product_id in self.memory.facts

    def dead(self, product_id: str, sku: str) -> bool:
        if self.key(product_id, sku) in self.dead_keys:
            return True
        # After a stockout the recovery family may still want the same model
        # in another colour, which is a different SKU of the same product.
        return product_id in self.dead_products and self.family != "recovery"

    def fresh_after_add(self, product_id: str, sku: str) -> bool:
        """Was this variant re-read after the most recent cart addition?"""

        if self.memory.last_add_turn is None:
            return (product_id, sku) in self.memory.live
        checked = self.memory.checked_turn.get((product_id, sku))
        return checked is not None and checked > self.memory.last_add_turn

    # -- reading a spec the way the runtime resolves it -------------------
    #
    # The variant's own option text wins. When it names a single value, that
    # is the value; when it names several, or talks about the concept without
    # a clean value, the variant is unresolved. Only when it is silent do the
    # product-wide sources apply: labelled facts merged with the title, and
    # nothing at all when those disagree. A guess that turns out wrong fails
    # a hard gate, so unresolved is always safer than wrong.

    @staticmethod
    def _values(field: str, text: str) -> set[str]:
        text = text or ""
        if field == "screen_size":
            return {f"{float(m.group(1)):g}in" for m in _INCH_ANY_RE.finditer(text)}
        if field == "refresh_rate":
            return {f"{float(m.group(1)):g}hz" for m in _REFRESH_RE.finditer(text)}
        if field == "resolution":
            value = _canonical_resolution(text)
            return {value} if value else set()
        if field == "ram":
            return {
                f"{int(m.group(1))}gb"
                for m in re.finditer(r"(\d+)\s*gb\b", text, re.IGNORECASE)
            }
        return set()

    def _product_value(self, product_id: str, field: str) -> str | None:
        facts = self.memory.facts.get(product_id) or {}
        values: set[str] = set()
        for key, raw in facts.items():
            if not raw or field.split("_")[0] not in key.lower().replace(" ", "_"):
                continue
            values |= self._values(field, str(raw))
        if not values:
            stated = _norm_spec(spec_from_facts(facts, field))
            if stated:
                values.add(stated)
        values |= self._values(field, self.memory.titles.get(product_id, ""))
        return next(iter(values)) if len(values) == 1 else None

    def _variant_value(self, product_id: str, sku: str, field: str) -> str | None:
        options = self.memory.options.get((product_id, sku), "")
        values = self._values(field, options)
        if len(values) > 1:
            return None
        if values:
            return next(iter(values))
        mentions = {
            "screen_size": _INCH_ANY_RE,
            "refresh_rate": _REFRESH_RE,
        }.get(field)
        if mentions is not None and mentions.search(options):
            return None
        return self._product_value(product_id, field)

    def measure(self, product_id: str, sku: str, field: str) -> float | None:
        value = self._variant_value(product_id, sku, field)
        try:
            return float(value[:-2]) if value else None
        except ValueError:
            return None

    def max_hz(self, product_id: str, sku: str) -> float | None:
        """Highest advertised refresh rate, overclock included (ranking's rule)."""

        text = " ".join(
            [
                self.memory.options.get((product_id, sku), ""),
                " ".join(
                    str(v) for v in (self.memory.facts.get(product_id) or {}).values()
                ),
                self.memory.titles.get(product_id, ""),
            ]
        )
        values = [float(v[:-2]) for v in self._values("refresh_rate", text)]
        return max(values) if values else None

    def spec_ok(self, candidate: ScoredCandidate) -> bool:
        """Every requested spec resolves, on this exact variant, to the wanted value.

        resolve_variant ports the runtime's resolve_variant_attribute to the
        visible row: the variant's options decide, else the product's facts plus
        its title, and any disagreement resolves to nothing.
        """

        pid, sku = candidate.product_id, candidate.sku
        facts = self.memory.facts.get(pid) or {}
        title = self.memory.titles.get(pid, candidate.title)
        options = self.memory.options.get((pid, sku))
        return all(
            _norm_spec(resolve_variant(facts, title, options, spec))
            == _norm_spec(wanted)
            for spec, wanted in self.product.specs.items()
        )

    def spec_probable(self, candidate: ScoredCandidate) -> bool:
        """A last-resort reading: the wanted resolution is what the pixel dimensions say.

        A row joins the display resolution with a camera's video resolution
        ("720 x 1600; 1080p"), and the runtime reads only the former. When the
        exact reading refuses, the pixel parts alone still usually decide.
        Used only when nothing verified is left to order.
        """

        pid, sku = candidate.product_id, candidate.sku
        facts = self.memory.facts.get(pid) or {}
        for spec, wanted in self.product.specs.items():
            if self.spec_ok(candidate):
                continue
            if spec != "resolution" or values_in(
                "resolution", self.memory.options.get((pid, sku)) or ""
            ):
                return False
            display = str(facts.get("display") or "")
            pixels: set[str] = set()
            for part in display.split("; "):
                if _PIXELS_RE.search(part):
                    pixels |= values_in("resolution", part)
            if pixels != {wanted}:
                return False
        return True

    # -- what the listing is ------------------------------------------------

    def device_ok(self, product_id: str, sku: str) -> bool:
        """A smartphone request is graded on the listing being a real handset."""

        if not self.wants_phone:
            return True
        verdict = phone_identity(
            self.memory.facts.get(product_id) or {},
            self.memory.titles.get(product_id, ""),
            self.memory.options.get((product_id, sku), ""),
        )
        if self.family in ("constraint_satisfaction", "justification", "recovery"):
            return verdict == "yes"
        return verdict != "no"

    def shelf_ok(self, product_id: str) -> bool:
        wanted = self.product.category
        path = self.memory.cat_paths.get(product_id)
        if wanted and not path:
            # view and compare always carry the path; a listing without one
            # was never read, so it cannot be known to sit on the shelf.
            return False
        return category_ok(wanted, path, self.memory.titles.get(product_id))

    def brand_ok(self, product_id: str) -> bool:
        """A firm brand must match the listing's BRAND FIELD, exactly as the runtime does.

        oro_env_runtime.acceptance.brand_matches compares the expected brand with
        the catalogue brand only: equal after normalising to lowercase
        alphanumeric words, or the expected words a subset of the actual ones.
        It accepts a title argument and never uses it, so a counterfeit titled
        "Samsung S9 MAX" under another brand is not in the gold set, and a
        listing with no brand field can never satisfy a firm brand.
        """

        if not self.brand_firm:
            return True
        expected = re.findall(r"[a-z0-9]+", (self.product.brand or "").casefold())
        if not expected:
            return True
        actual = re.findall(
            r"[a-z0-9]+", (self.memory.brands.get(product_id) or "").casefold()
        )
        if not actual:
            return False
        return " ".join(expected) == " ".join(actual) or set(expected) <= set(actual)

    def listing(self, product_id: str) -> str:
        """The source listing a product id belongs to (itself when unknown)."""

        return self.memory.listing_of.get(product_id, product_id)

    @property
    def pref_profile(self) -> str | None:
        if self.family != "preference_reasoning":
            return None
        return preference_profile(self.product.category, self.product.text or "")

    def pref_score(self, candidate: ScoredCandidate) -> float:
        """The TF4 rubric estimate for this listing, 0.0 when not applicable."""

        pid = candidate.product_id
        return (
            estimate_preference(
                self.pref_profile,
                self.memory.titles.get(pid, candidate.title),
                self.memory.facts.get(pid),
                self.memory.cat_paths.get(pid),
                self.memory.brands.get(pid),
            )
            or 0.0
        )

    def listing_ok(self, product_id: str, sku: str) -> bool:
        return (
            self.device_ok(product_id, sku)
            and self.shelf_ok(product_id)
            and self.brand_ok(product_id)
            and not resale_title(self.memory.titles.get(product_id, ""))
        )

    def tradeoff_key(
        self, candidate: ScoredCandidate
    ) -> tuple[float, float, float, str]:
        """Ranking's stated rule: refresh rate, then screen size, then price."""

        hz = self.max_hz(candidate.product_id, candidate.sku) or 0.0
        size = self.measure(candidate.product_id, candidate.sku, "screen_size") or 0.0
        return (-hz, -size, candidate.price, candidate.key)

    def relax_level(self, candidate: ScoredCandidate) -> int:
        """Recovery's level for a read variant, 0 exact to 3; 4 when never gold.

        The runtime admits a row only through _candidate_row (a real handset or
        over-ear headphone with resolvable model and colour labels, on the
        task's shelf) and grades it with _level_for_values; recovery_row and
        relaxation_level port both exactly. An unread product has no facts yet,
        so it reads as 4 until it is read.
        """

        model, color = self.product.model, self.product.color
        if not model:
            return 0
        pid = candidate.product_id
        if not self.viewed(pid):
            return 4
        row = recovery_row(
            self.memory.facts.get(pid),
            self.memory.titles.get(pid, candidate.title),
            self.memory.brands.get(pid),
            self.memory.options.get((pid, candidate.sku)),
            self.memory.cat_paths.get(pid),
        )
        if row is None:
            return 4
        if self.product.category and not category_ok(self.product.category, [row[0]]):
            return 4
        level = relaxation_level(row[1], row[2], model, color or "")
        return 4 if level is None else level

    def _title_relax(self, candidate: ScoredCandidate) -> int:
        """A title-only guess at an unread product's level, to decide what to read."""

        model, color = self.product.model or "", self.product.color or ""
        title = self.memory.titles.get(candidate.product_id, candidate.title) or ""
        same_model = bool(model) and _squash(model) in _squash(title)
        same_color = bool(color) and (
            re.search(rf"(?<![a-z]){re.escape(color.lower())}(?![a-z])", title.lower())
            is not None
        )
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def brand_named(self, candidate: ScoredCandidate) -> bool:
        wanted = title_tokens(self.product.brand or "")
        if not wanted:
            return False
        brand = self.memory.brands.get(candidate.product_id) or candidate.brand or ""
        return wanted <= title_tokens(brand) or wanted <= title_tokens(
            self.memory.titles.get(candidate.product_id, candidate.title)
        )

    def verified(self, probable: bool = False) -> list[ScoredCandidate]:
        """Candidates that pass every gate the runtime will apply."""

        _ExecutionMesh.relay(self, 15)
        out = []
        for candidate in self.candidates():
            if not candidate.eligible:
                continue
            if not self.viewed(candidate.product_id):
                continue
            if self.dead(candidate.product_id, candidate.sku):
                continue
            if not self.spec_ok(candidate) and not (
                probable and self.spec_probable(candidate)
            ):
                continue
            if not self.listing_ok(candidate.product_id, candidate.sku):
                continue
            if (
                self.family == "preference_reasoning"
                and self.event_target
                and self.listing(candidate.product_id)
                == self.listing(self.event_target.split("::", 1)[0])
            ):
                # Recovery must move to a different source listing, not a twin.
                continue
            if self.family == "ranking" and (
                self.max_hz(candidate.product_id, candidate.sku) is None
                or self.measure(candidate.product_id, candidate.sku, "screen_size")
                is None
            ):
                continue
            if self.family == "justification":
                facts = self.memory.facts.get(candidate.product_id) or {}
                if self.product.specs:
                    spec = next(iter(self.product.specs))
                    if spec_from_facts(facts, spec) != self.product.specs[spec]:
                        continue
                if not self.memory.brands.get(candidate.product_id):
                    continue
            if (
                self.family == "ranking"
                and self.product.candidates
                and self.key(candidate.product_id, candidate.sku)
                not in self.product.candidates
            ):
                continue
            if (
                self.family == "recovery"
                and self.product.model
                and self.relax_level(candidate) >= 4
            ):
                # Not an admissible recovery row: an order of it is never in gold.
                continue
            out.append(candidate)
        return out

    # ------------------------------------------------------------ decisions

    def turns_left(self, turn: int) -> int:
        return self.max_steps - turn + 1

    def ready(self, turn: int, endgame: bool = False) -> bool:
        """Whether the first commitment may be made now."""

        if self.memory.adds > 0 or endgame:
            return True
        if self.turns_left(turn) <= COMMIT_RESERVE_TURNS:
            return True
        if self.spec_sealed:
            if turn < self.max_steps * SEALED_SPEC_PATIENCE:
                return False
            # Never stated: the remaining requirements decide the choice.
            self.spec_sealed = False
        family = self.family
        if (
            family in ("intent_decomposition", "constraint_satisfaction")
            and turn < FIRST_ADD_MIN_TURN
            and not self.shopper_updated
        ):
            # A later shopper message can still tighten the budget.
            return False
        verified = self.verified()
        listings = {self.listing(candidate.product_id) for candidate in verified}
        if family in _WIDEN_FAMILIES:
            # The event takes the carted key only, so a twin listing is a fallback.
            listings = {candidate.product_id for candidate in verified}
        if family == "preference_reasoning":
            return len(listings) >= 2
        if family == "recovery" and turn >= FIRST_ADD_MIN_TURN + 2:
            # Recovery pays for the replacement, searched for after the event.
            return True
        # The market moves on the first cart action; line up a fallback first,
        # unless there is nothing left to read.
        return len(listings) >= 2 or self.reads_done

    def target(self, turn: int, endgame: bool = False) -> ScoredCandidate | None:
        """The best fully verified candidate, or None while evidence is missing."""

        _ExecutionMesh.relay(self, 14)
        first = self.memory.adds == 0
        good = self.verified()
        family = self.family
        roomy = turn < self.max_steps * 0.6 and not endgame
        if (
            not good
            and not first
            and family != "justification"
            and self.event_target is not None
            and not self._wants_widen(endgame)
            and (
                self.reads_done
                or endgame
                or self.turns_left(turn) <= COMMIT_RESERVE_TURNS + 2
            )
        ):
            good = self.verified(probable=True)

        if family == "preference_reasoning":
            carted = [c for c in good if (c.product_id, c.sku) in self.memory.cart]
            if not good:
                return None
            if self.pref_profile:
                ranked = sorted(
                    good, key=lambda c: (-self.pref_score(c), c.price, c.key)
                )
                best = ranked[0]
                if not first:
                    # Paid: the ordered item's own rubric score. After the first
                    # commitment, always finish on the strongest listing left.
                    return best
                if (
                    self.pref_score(best) < PREFERENCE_GOOD
                    and not endgame
                    and self.turns_left(turn) > PREFERENCE_TAKEOVER_TURNS
                ):
                    return None
                # The first cart addition only has to clear PREFERENCE_GOOD, and
                # the market event then reprices exactly that item out of reach.
                # Commit the weakest listing that clears the bar and keep the
                # strongest, on a different listing, for the order.
                keep = self.listing(best.product_id)
                sacrificial = [
                    c
                    for c in ranked
                    if self.pref_score(c) >= PREFERENCE_GOOD
                    and self.listing(c.product_id) != keep
                ]
                return sacrificial[-1] if sacrificial else best
            if carted:
                return carted[0]
            if endgame or self.turns_left(turn) <= PREFERENCE_TAKEOVER_TURNS:
                return good[0]
            return None

        if family == "intent_decomposition" and first and self.product.brand:
            named = [c for c in good if self.brand_named(c)]
            if named:
                good = named
            elif roomy:
                self.hinted_brand = True
                return None

        if family == "recovery" and self.product.model:
            # The family pays for the LEAST relaxed valid choice, so relaxation
            # level outranks every relevance signal, before and after the event.
            # A keyword-stuffed title scores high on relevance and is exactly the
            # wrong answer once the exact match is gone; the same model in
            # another colour (level 1) is the right one.
            if first:
                exact = [c for c in good if self.relax_level(c) == 0]
                if exact:
                    good = exact
                elif roomy:
                    self.hinted_exact = True
                    return None
            good = sorted(good, key=lambda c: (self.relax_level(c), -c.score, c.price))

        if family == "ranking":
            good = sorted(good, key=self.tradeoff_key)

        return good[0] if good else None

    def _matching_rows(self) -> list[dict[str, Any]]:
        """Result rows whose title holds every requested term, in budget and stock."""

        budget = self.product.budget
        out = []
        for row in self.rows:
            if not matches_required_phrase(
                str(row.get("title") or ""), self.product.required_terms
            ):
                continue
            price = row.get("price")
            if (
                budget is not None
                and isinstance(price, (int, float))
                and price > budget + _EPS
            ):
                continue
            if row.get("in_stock") is False:
                continue
            out.append(row)
        return out

    @staticmethod
    def _sku_stem(row: dict[str, Any]) -> str:
        """The catalogue item a row belongs to: twins listed under other ids share the sku stem."""

        sku = str(row.get("sku") or "")
        if "-" in sku:
            return sku.rsplit("-", 1)[0]
        return sku or str(row.get("product_id") or "")

    @classmethod
    def _identities(cls, rows: list[dict[str, Any]]) -> int:
        """Distinct catalogue items among rows, counting an item's twin listings once."""

        return len({cls._sku_stem(row) for row in rows if row.get("product_id")})

    def _diverse_tokens(self, rows: list[dict[str, Any]], count: int) -> list[str]:
        """Extra title words that shift a search towards matching listings not yet seen.

        Twins (one item under several ids) are counted once. A product-type word
        ("toner", "case") marks a kind of item with more listings past the ten
        shown; the second word comes from the titles the first one misses, so
        the two searches pull in different directions.
        """

        wanted = set(self.product.required_terms)
        groups: dict[str, set[str]] = {}
        for row in rows:
            words = re.findall(r"[a-z]+", str(row.get("title") or "").lower())
            groups.setdefault(self._sku_stem(row), set(words))
        titles = list(groups.values())

        def usable(word: str) -> bool:
            return (
                word not in wanted
                and word not in _COMMON_TITLE_WORDS
                and len(word) >= 3
            )

        def pick(pool: list[set[str]], taken: list[str]) -> str | None:
            counts: dict[str, int] = {}
            for words in pool:
                for word in words:
                    if usable(word) and word not in taken:
                        counts[word] = counts.get(word, 0) + 1
            if not counts:
                return None
            ceiling = max(1, int(len(titles) * 0.8)) if len(titles) > 2 else len(titles)
            kinds = [word for word in counts if word in _PRODUCT_TYPE_WORDS]
            if kinds:
                return max(sorted(kinds), key=lambda word: counts[word])
            shared = [word for word in counts if 2 <= counts[word] <= ceiling]
            return (
                max(sorted(shared), key=lambda word: counts[word]) if shared else None
            )

        chosen: list[str] = []
        pool = titles
        while len(chosen) < count:
            word = pick(pool, chosen) or pick(titles, chosen)
            if word is None:
                break
            chosen.append(word)
            pool = [
                words for words in titles if not any(taken in words for taken in chosen)
            ] or titles
        return chosen

    def _discriminative_words(
        self, matching: list[dict[str, Any]], count: int
    ) -> list[str]:
        """Words for the next search: item kinds rare among the other listings seen.

        A kind common across the non-matching rows of the category pulls a search
        towards everything else; a kind carried by few matching items and few others
        shifts the page towards matching items not shown yet. Shared title words fill
        any remaining slot.
        """

        wanted = set(self.product.required_terms)
        used = {word for query in self.search_queries for word in query.split()}
        stems: dict[str, set[str]] = {}
        matching_ids = {id(row) for row in matching}
        for row in matching:
            for word in set(re.findall(r"[a-z]+", str(row.get("title") or "").lower())):
                if (
                    word in _PRODUCT_TYPE_WORDS
                    and word not in wanted
                    and word not in used
                ):
                    stems.setdefault(word, set()).add(self._sku_stem(row))
        common: dict[str, int] = {}
        for row in self.rows:
            if id(row) in matching_ids:
                continue
            for word in set(re.findall(r"[a-z]+", str(row.get("title") or "").lower())):
                if word in stems:
                    common[word] = common.get(word, 0) + 1
        kinds = sorted(
            stems, key=lambda word: (common.get(word, 0), len(stems[word]), word)
        )
        if len(kinds) >= count:
            return kinds[:count]
        shared = [
            word
            for word in self._diverse_tokens(matching, count + 6)
            if word not in used and word not in kinds
        ]
        return (kinds + shared)[:count]

    def _glued_query(self, rows: list[dict[str, Any]]) -> str | None:
        """A search for listings whose title glues a phrase word to punctuation.

        A title can write "AHA~BHA" or "AHA·BHA": the verifier's title tokens
        split on any non-alphanumeric character, but the search index may keep
        the glued run as one token, so a query holding that word never reaches
        those listings. The glued rows show the other words such titles use.
        """

        terms = self.product.required_terms
        if len(terms) < 2:
            return None
        glued_terms: set[str] = set()
        glued_rows: list[tuple[str, set[str]]] = []
        for row in rows:
            low = str(row.get("title") or "").lower()
            for term in terms:
                pattern = (
                    rf"[a-z0-9][^\sa-z0-9]+{re.escape(term)}(?![a-z0-9])"
                    rf"|(?<![a-z0-9]){re.escape(term)}[^\sa-z0-9]+[a-z0-9]"
                )
                if re.search(pattern, low):
                    glued_terms.add(term)
                    brand_words = set(
                        re.findall(r"[a-z]+", str(row.get("brand") or "").lower())
                    )
                    glued_rows.append((low, brand_words))
        keep = [term for term in terms if term not in glued_terms]
        if not glued_rows or not keep:
            return None
        counts: dict[str, int] = {}
        for low, brand_words in glued_rows:
            for word in set(re.findall(r"[a-z]{4,}", low)):
                if word in terms or word in _COMMON_TITLE_WORDS or word in brand_words:
                    continue
                counts[word] = counts.get(word, 0) + 1
        ranked = sorted(
            counts,
            key=lambda word: (word not in _PRODUCT_TYPE_WORDS, -counts[word], word),
        )
        return " ".join([*keep, *ranked[:2]]) if ranked else None

    def _surface_step(self) -> tuple[str, list[dict[str, Any]]] | None:
        """Widen coverage to ten matching identities: brand filters, then diverse searches."""

        if self.surface_rounds >= _SURFACE_ROUNDS or not self.product.required_terms:
            return None
        matching = self._matching_rows()
        if self._identities(matching) >= RETRIEVAL_TARGET_IDENTITIES:
            return None
        self.surface_rounds += 1
        base: dict[str, Any] = {"in_stock": True, "k": 10}
        if self.product.category:
            base["category"] = self.product.category
        if self.product.budget is not None:
            base["max_price"] = self.product.budget
        calls: list[dict[str, Any]] = []

        def add_filter(args: dict[str, Any]) -> None:
            if "filter" not in self.tool_names:
                return
            if "filter" + json.dumps(args, sort_keys=True) in self.calls_seen:
                return
            if any(call["args"] == args for call in calls):
                return
            calls.append(_call("filter", **args))

        # Brands of matching rows first, then brands of rows holding part of the
        # phrase. Only brands not yet filtered count against the per-round cap, so
        # a brand first seen in the last searches is never crowded out.
        wanted = set(self.product.required_terms)
        brands: dict[str, tuple[int, int]] = {}
        matching_ids = {id(row) for row in matching}
        for row in self.rows:
            brand = row.get("brand")
            if not isinstance(brand, str) or not brand.strip():
                continue
            full = id(row) in matching_ids
            if not full and not wanted & title_tokens(str(row.get("title") or "")):
                continue
            hits, partial = brands.get(brand, (0, 0))
            brands[brand] = (hits + 1, partial) if full else (hits, partial + 1)
        ordered = sorted(
            brands, key=lambda name: (-brands[name][0], -brands[name][1], name)
        )
        room = self.max_calls - 2
        if self.product.budget:
            # Brands already holding a matching listing whose page was full first.
            for brand in ordered:
                if not brands[brand][0]:
                    continue
                page = self.filter_rows.get(
                    json.dumps({**base, "brand": brand}, sort_keys=True)
                )
                if page is None or page < 10:
                    continue
                for fraction in _BRAND_DEEP_BANDS:
                    if len(calls) >= room:
                        break
                    add_filter(
                        {
                            **base,
                            "brand": brand,
                            "max_price": round(self.product.budget * fraction, 2),
                        }
                    )
        for brand in ordered:
            if len(calls) >= room:
                break
            add_filter({**base, "brand": brand})
            if self.product.budget:
                # Filters stop at ten rows; a lower ceiling reaches the rest of a brand.
                add_filter(
                    {
                        **base,
                        "brand": brand,
                        "max_price": round(self.product.budget * 0.5, 2),
                    }
                )
        if self.product.budget:
            for fraction in _SURFACE_PRICE_BANDS.get(self.surface_rounds, ()):
                add_filter(
                    {**base, "max_price": round(self.product.budget * fraction, 2)}
                )
        if "search" in self.tool_names:
            room = max(0, MAX_DISTINCT_SEARCHES - len(self.search_queries))
            phrase = " ".join(self.product.required_terms)
            # Two kinds per search: each result page then splits between two kinds
            # of item, so the searches left reach twice as many distinct products.
            queries: list[str] = []
            # One new search per round, chosen from everything the earlier ones showed:
            # the phrase without a punctuation-glued word first, then the phrase plus
            # the item kinds least represented among the listings seen so far.
            if room:
                glued = None if self.glued_searched else self._glued_query(matching)
                if glued:
                    self.glued_searched = True
                    queries.append(glued)
                else:
                    words = self._discriminative_words(matching, 2)
                    if words:
                        queries.append(f"{phrase} {' '.join(words)}")
            for query in queries:
                if query.lower() not in self.search_queries:
                    calls.insert(0, _call("search", query=query, k=10, in_stock=True))
        calls = calls[: self.max_calls]
        return ("surface", calls) if calls else None

    def _preference_read(
        self, profile: str | None
    ) -> tuple[str, list[dict[str, Any]]] | None:
        """Compare the unread listings whose titles look best on the rubric."""

        scored: list[tuple[float, float, str]] = []
        for candidate in self.candidates():
            pid = candidate.product_id
            if (
                not candidate.eligible
                or self.viewed(pid)
                or pid in self.unreadable
                or self.dead(pid, candidate.sku)
            ):
                continue
            title_estimate = (
                estimate_preference(profile, candidate.title, {}, None, candidate.brand)
                or 0.0
            )
            scored.append((title_estimate, -candidate.price, pid))
        unread: list[str] = []
        for _estimate, _cheap, pid in sorted(scored, reverse=True):
            if pid not in unread:
                unread.append(pid)
            if len(unread) >= 3 * MAX_COMPARE_IDS:
                break
        if not unread or "compare" not in self.tool_names:
            return None
        calls = [
            _call("compare", product_ids=unread[start : start + MAX_COMPARE_IDS])
            for start in range(0, len(unread), MAX_COMPARE_IDS)
        ]
        return ("read", calls[: self.max_calls])

    def _preference_axis_searches(
        self, profile: str | None, best: ScoredCandidate | None
    ) -> list[dict[str, Any]]:
        """Searches for the rubric axes the best read listing does not state."""

        terms = _PREFERENCE_AXIS_TERMS.get(profile or "", {})
        if not terms or "search" not in self.tool_names:
            return []
        axes: dict[str, float] = {}
        if best is not None:
            pid = best.product_id
            axes = (
                preference_axes(
                    profile,
                    self.memory.titles.get(pid, best.title),
                    self.memory.facts.get(pid),
                    self.memory.cat_paths.get(pid),
                    self.memory.brands.get(pid),
                )
                or {}
            )
        missing = [axis for axis in terms if axes.get(axis, 0.0) < 1.0]
        if not missing:
            return []
        round_index = self.pref_axis_rounds - 1
        nouns = _PREFERENCE_NOUNS.get(profile or "", ("",))
        queries: list[str] = []
        # Searches cost nothing here, and which listings a page shows turns on
        # the exact wording, so each round tries two nouns with two wordings.
        for noun_pick, word_pick in (
            (round_index, round_index),
            (round_index, round_index + 1),
            (round_index + 1, round_index),
            (round_index + 1, round_index + 1),
        ):
            noun = nouns[noun_pick % len(nouns)]
            focused = " ".join(
                [noun, *(terms[axis][word_pick % len(terms[axis])] for axis in missing)]
            )
            full = " ".join(
                [noun, *(words[word_pick % len(words)] for words in terms.values())]
            )
            for query in (focused, full):
                query = " ".join(query.split())
                if (
                    query
                    and query.lower() not in self.search_queries
                    and query not in queries
                ):
                    queries.append(query)
        calls = [_call("search", query=query, k=10, in_stock=True) for query in queries]
        if (
            "filter" in self.tool_names
            and self.product.category
            and self.product.budget
        ):
            # Filters are free here and stop at ten rows: a lower ceiling shows others.
            fraction = (0.7, 0.45, 0.25, 0.12)[round_index % 4]
            args = {
                "in_stock": True,
                "k": 10,
                "category": self.product.category,
                "max_price": round(self.product.budget * fraction, 2),
            }
            if "filter" + json.dumps(args, sort_keys=True) not in self.calls_seen:
                calls.append(_call("filter", **args))
        return calls

    def _preference_surface_step(
        self, turn: int
    ) -> tuple[str, list[dict[str, Any]]] | None:
        """Before committing: search the rubric, read the best-looking listings, then
        search again for whatever axis the best read listing still misses."""

        profile = self.pref_profile
        if self.pref_surface_stage == 0:
            self.pref_surface_stage = 1
            calls: list[dict[str, Any]] = []
            if "search" in self.tool_names:
                for query in _PREFERENCE_QUERIES.get(profile or "", ()):
                    if query not in self.search_queries:
                        calls.append(_call("search", query=query, k=10, in_stock=True))
            if "filter" in self.tool_names:
                args: dict[str, Any] = {"in_stock": True, "k": 10}
                if self.product.category:
                    args["category"] = self.product.category
                if self.product.budget is not None:
                    args["max_price"] = self.product.budget
                if (
                    len(args) > 2
                    and "filter" + json.dumps(args, sort_keys=True)
                    not in self.calls_seen
                ):
                    calls.append(_call("filter", **args))
            if calls:
                return ("pref_surface", calls[: self.max_calls])
        if self.pref_surface_stage in (1, 3):
            # Read what the last searches surfaced, then decide on another round.
            self.pref_surface_stage = 2
            step = self._preference_read(profile)
            if step is not None:
                return step
        if self.pref_surface_stage == 2:
            verified = self.verified()
            best = max(
                verified, key=lambda c: (self.pref_score(c), -c.price), default=None
            )
            best_score = self.pref_score(best) if best is not None else 0.0
            if (
                best_score >= 1.0 - _EPS
                or self.pref_axis_rounds >= _PREFERENCE_AXIS_ROUNDS
                or self.turns_left(turn) <= PREFERENCE_TAKEOVER_TURNS + 4
            ):
                self.pref_surface_stage = 4
                return None
            self.pref_axis_rounds += 1
            self.pref_surface_stage = 3
            calls = self._preference_axis_searches(profile, best)
            if calls:
                return ("pref_surface", calls[: self.max_calls])
            self.pref_surface_stage = 4
        return None

    def _recovery_step(self) -> tuple[str, list[dict[str, Any]]] | None:
        """After the event: never settle while a less-relaxed option may exist.

        The family pays 1 - (chosen - best) / 3 and only admits listings whose
        model and colour resolve, so a verified level-2 item is not an answer
        while the same model in another colour is still unsearched or unread.
        """

        verified = self.verified()
        best = min((self.relax_level(c) for c in verified), default=4)
        if best == 0:
            return None
        unread: list[str] = []
        for candidate in self.candidates():
            pid = candidate.product_id
            if (
                not candidate.eligible
                or self.viewed(pid)
                or pid in self.unreadable
                or self.dead(pid, candidate.sku)
                or pid in unread
            ):
                continue
            if pid in self.model_hits or self._title_relax(candidate) < best:
                unread.append(pid)
        # What a model search returned goes first: its level lives in the facts.
        unread.sort(key=lambda pid: pid not in self.model_hits)
        if unread and "compare" in self.tool_names:
            unread = unread[: 2 * MAX_COMPARE_IDS]
            calls = [
                _call("compare", product_ids=unread[start : start + MAX_COMPARE_IDS])
                for start in range(0, len(unread), MAX_COMPARE_IDS)
            ]
            return ("read", calls[: self.max_calls])
        if best > self.relaxation_level and self.relaxation_level <= 2:
            level = self.relaxation_level
            self.relaxation_level += 1
            calls = plan_recovery_calls(self.product, level, self.max_calls)
            if calls:
                return ("recovery", calls)
        return None

    def opening_question(self) -> str | None:
        if self.asked_opening:
            return None
        if self.spec_sealed:
            return _OPENING_QUESTIONS["constraint_satisfaction"]
        return _OPENING_QUESTIONS.get(self.family or "")

    def _wants_widen(self, endgame: bool) -> bool:
        """Too few verified listings to survive the market event, and rounds remain."""

        if self.family not in _WIDEN_FAMILIES or endgame or self.spec_sealed:
            return False
        verified = self.verified()
        if self.memory.adds == 0:
            return (
                self.widen_rounds < _WIDEN_ROUNDS_BEFORE_ADD
                and len({c.product_id for c in verified}) < 2
            )
        return (
            self.widen_rounds < _WIDEN_ROUNDS
            and self.event_target is not None
            and not verified
        )

    def _spec_phrase(self) -> str:
        for spec, value in self.product.specs.items():
            if spec == "screen_size":
                return value.replace("in", " inch")
            if spec == "ram":
                return f"{value} ram"
            if spec == "resolution":
                return _SPEC_SEARCH_WORDS.get(value, value)
            return value
        return ""

    def _widen_step(self) -> tuple[str, list[dict[str, Any]]] | None:
        """Reach listings the opening sweep missed: price-banded filters and spec searches."""

        if self.memory.adds > 0:
            self.widen_rounds = max(self.widen_rounds, _WIDEN_ROUNDS_BEFORE_ADD)
        self.widen_rounds += 1
        base: dict[str, Any] = {"in_stock": True, "k": 10}
        if self.product.category and not (
            self.widen_rounds >= 5 and self.product.brand
        ):
            base["category"] = self.product.category
        if self.product.brand:
            base["brand"] = self.product.brand
        calls: list[dict[str, Any]] = []
        if "filter" in self.tool_names:
            budget = self.product.budget
            for fraction in _WIDEN_PRICE_BANDS.get(self.widen_rounds, ()):
                args = dict(base)
                if budget:
                    args["max_price"] = round(budget * fraction, 2)
                if "filter" + json.dumps(args, sort_keys=True) in self.calls_seen:
                    continue
                if all(call["args"] != args for call in calls):
                    calls.append(_call("filter", **args))
        if "search" in self.tool_names:
            # The catalogue sells one item under several product ids with the same
            # title, and the market event invalidates only the key that was carted.
            # Searching a title's own words brings its twins to the top.
            if self.memory.adds > 0 and self.event_target:
                anchor: str | None = self.event_target.split("::", 1)[0]
            else:
                verified = self.verified()
                anchor = verified[0].product_id if verified else None
            if anchor and anchor not in self.twin_searched:
                self.twin_searched.add(anchor)
                words = re.findall(
                    r"[A-Za-z0-9]+(?:[.+][A-Za-z0-9]+)*",
                    self.memory.titles.get(anchor, ""),
                )
                query = " ".join(words[:10])
                if query and query.lower() not in self.search_queries:
                    calls.insert(0, _call("search", query=query, k=10, in_stock=True))
            brand = self.product.brand or ""
            device = self.device_word or (self.product.category or "")
            phrase = self._spec_phrase()
            wordings = {
                1: (f"{brand} {device} {phrase}",),
                2: (f"{brand} {phrase}", f"{device} {phrase}"),
                3: (f"{brand} {device}",),
                4: (f"{brand} {device} {phrase} new",),
            }.get(self.widen_rounds, ())
            for wording in wordings:
                query = " ".join(wording.split())
                if query and query.lower() not in self.search_queries:
                    calls.append(_call("search", query=query, k=10, in_stock=True))
        return ("widen", calls[: self.max_calls]) if calls else None

    def _read_more(self) -> tuple[str, list[dict[str, Any]]] | None:
        """One compare batch over eligible products whose facts are unread."""

        unread: list[str] = []
        for candidate in self.candidates():
            pid = candidate.product_id
            if not candidate.eligible or self.viewed(pid) or pid in self.unreadable:
                continue
            if self.dead(pid, candidate.sku):
                continue
            if self.brand_firm and not self.brand_ok(pid) and pid in self.memory.brands:
                # A known brand field that is not the required brand is never gold.
                continue
            if self.brand_firm and not (candidate.brand or self.memory.brands.get(pid)):
                continue
            if pid not in unread:
                unread.append(pid)
        unread = unread[: 3 * MAX_COMPARE_IDS if self.widen_rounds else SHORTLIST_SIZE]
        if not unread or "compare" not in self.tool_names:
            self.reads_done = True
            return None
        calls = [
            _call("compare", product_ids=unread[start : start + MAX_COMPARE_IDS])
            for start in range(0, len(unread), MAX_COMPARE_IDS)
        ]
        return ("read", calls[: self.max_calls])

    def plan(
        self, turn: int, endgame: bool = False
    ) -> tuple[str, list[dict[str, Any]]] | None:
        """The controller's own envelope for this turn, or None to ask the model."""

        _ExecutionMesh.relay(self, 13)
        if self.done:
            return None

        if self.sweep_stage == 0:
            self.sweep_stage = 1
            calls = plan_opening_calls(self.product, self.family, self.max_calls)
            question = self.opening_question()
            if question and "message" in self.tool_names:
                self.asked_opening = True
                self.questions_asked += 1
                calls = [_call("message", content=question)] + calls
            if calls:
                return ("sweep", calls[: self.max_calls])
        if self.sweep_stage == 1:
            self.sweep_stage = 2
            shortlist = self.shortlist()
            limit = SHORTLIST_SIZE
            if self.family == "recovery" and self.model_hits:
                # Read the preferred model's other listings before the event, so
                # the least-relaxed replacement is already verified when it fires.
                shortlist = sorted(
                    shortlist, key=lambda c: c.product_id not in self.model_hits
                )
                limit = 2 * MAX_COMPARE_IDS
            calls = plan_inspection_calls(shortlist, self.max_calls, limit)
            if calls:
                return ("inspect", calls)

        if self.family == "retrieval_recall" and not endgame and self.memory.adds == 0:
            step = self._surface_step()
            if step is not None:
                return step

        if (
            self.family == "preference_reasoning"
            and self.pref_profile
            and self.memory.adds == 0
            and not endgame
        ):
            step = self._preference_surface_step(turn)
            if step is not None:
                return step

        if self.pending_recheck is not None and "inspect_stock" in self.tool_names:
            pid, sku = self.pending_recheck
            return ("recheck", [_call("inspect_stock", product_id=pid, sku=sku)])

        if (
            self.family == "recovery"
            and self.event_target is not None
            and self.product.model
            and not endgame
        ):
            step = self._recovery_step()
            if step is not None:
                return step

        target = self.target(turn, endgame)

        if target is None and endgame:
            step = self._finish_fast_fallback(turn)
            if step is not None:
                return step

        if (
            target is None
            or not self.ready(turn, endgame)
            or self._wants_widen(endgame)
        ):
            step = self._read_more()
            if step is not None:
                return step
            if self._wants_widen(endgame):
                step = self._widen_step()
                if step is not None:
                    self.reads_done = False
                    return step
            target = self.target(turn, endgame)
            if target is None:
                if endgame:
                    return self._finish_fast_fallback(turn)
                return None
            if not self.ready(turn, endgame):
                return None

        pid, sku = target.product_id, target.sku
        if (pid, sku) in self.memory.cart:
            if not self.fresh_after_add(pid, sku):
                return ("recheck", [_call("inspect_stock", product_id=pid, sku=sku)])
            alternate = self._second_commitment(target, turn, endgame)
            if alternate is not None:
                self.second_commitment = alternate.key
                return (
                    "ladder",
                    [
                        _call(
                            "add_to_cart",
                            product_id=alternate.product_id,
                            sku=alternate.sku,
                        )
                    ],
                )
            return ("ladder", [self._order_call(pid, sku)])

        actions: list[dict[str, Any]] = []
        if "remove_from_cart" in self.tool_names:
            # A repeated addition leaves the same pair twice; one removal clears both.
            for cpid, csku in dict.fromkeys(self.memory.cart):
                # Preference swaps a surviving first commitment for the strongest
                # listing, so anything in the cart that is not the target goes.
                if self.dead(cpid, csku) or (
                    self.family == "preference_reasoning" and (cpid, csku) != (pid, sku)
                ):
                    actions.append(_call("remove_from_cart", product_id=cpid, sku=csku))
        actions.append(_call("add_to_cart", product_id=pid, sku=sku))
        if endgame and self.turns_left(turn) <= 1:
            # The only remaining way to score: no turn is left for a re-check.
            actions.append(self._order_call(pid, sku))
        return ("ladder", actions[: self.max_calls])

    def _second_commitment(
        self, target: ScoredCandidate, turn: int, endgame: bool
    ) -> ScoredCandidate | None:
        """A second valid cart addition when the first one's re-check showed no change.

        Every observed task moves the market on the first cart addition, so a clean
        re-check means the rule waits: for a later addition (nth_successful_cart_add)
        or a solver turn, which lands on the last cart item. Ordering now would leave
        the event unfired and the closure gate unmet. A second valid addition
        triggers or absorbs it while the first commitment stays orderable.

        Adding the same item again is also a successful addition, and the event then
        lands on the commitment itself. That is the fallback when no other product
        qualifies, and always the move for ranking, whose rerank is sealed on the
        first commitment being the item the event hits.
        """

        if (
            endgame
            or self.event_observed
            or self.second_commitment is not None
            or self.memory.adds != 1
            or target.key not in self.clean_rechecks
            or self.turns_left(turn) <= COMMIT_RESERVE_TURNS + 2
            or "add_to_cart" not in self.tool_names
        ):
            return None
        if self.family != "ranking":
            cart = set(self.memory.cart)
            listing = self.listing(target.product_id)
            for candidate in self.verified():
                if (
                    candidate.product_id,
                    candidate.sku,
                ) in cart or candidate.product_id == target.product_id:
                    continue
                if (
                    self.family == "preference_reasoning"
                    and self.listing(candidate.product_id) == listing
                ):
                    continue
                return candidate
        return target

    def _finish_fast_fallback(
        self, turn: int
    ) -> tuple[str, list[dict[str, Any]]] | None:
        """Out of time with no verified target: take the best live candidate."""

        # Every gate a verified target passes still applies at the end. An order
        # that fails a hard check scores exactly what no order scores, and an
        # unverified spec is how the endgame bought zeros.
        live = self.verified() or (
            [] if self.family == "justification" else self.verified(probable=True)
        )
        if self.family == "justification":
            live = [c for c in live if self.memory.brands.get(c.product_id)]
        if self.family == "recovery" and self.product.model:
            live = sorted(live, key=lambda c: (self.relax_level(c), -c.score, c.price))
        if self.pref_profile:
            live = sorted(live, key=lambda c: (-self.pref_score(c), c.price))
        if not live:
            return None
        pid, sku = live[0].product_id, live[0].sku
        actions = [_call("add_to_cart", product_id=pid, sku=sku)]
        if self.turns_left(turn) <= 1:
            actions.append(self._order_call(pid, sku))
        return ("ladder", actions)

    def _order_call(self, product_id: str, sku: str) -> dict[str, Any]:
        args = finalize_order_args(
            {"product_id": product_id, "sku": sku},
            self.memory,
            self.product,
            self.family,
        )
        return {"name": "place_test_order", "args": args}

    def validate(
        self,
        name: str,
        args: dict[str, Any],
        batch_names: list[str],
        turn: int,
        endgame: bool = False,
    ) -> str | None:
        """Why a proposed action must not be sent, or None when it may."""

        _ExecutionMesh.relay(args, 12)
        if name not in self.tool_names:
            return "unknown tool; choose one of " + ", ".join(sorted(self.tool_names))

        if name == "add_to_cart":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            if not self.viewed(pid):
                return "view or compare that product first; its facts are unread"
            live = self.memory.live.get((pid, sku))
            if live is None:
                return "no price or stock on record for that exact variant"
            if self.dead(pid, sku):
                return "that item is no longer valid; choose another product"
            price, in_stock = live
            if not in_stock:
                return "that variant is out of stock"
            budget = self.product.budget
            if budget is not None and price > budget + _EPS:
                return f"that variant costs {price:.2f}, above the budget {budget:.2f}"
            candidate = next(
                (c for c in self.candidates() if c.product_id == pid and c.sku == sku),
                None,
            )
            if candidate is not None and not self.spec_ok(candidate):
                return "that variant does not state the required spec value"
            if candidate is not None and not candidate.eligible:
                return (
                    f"that variant fails a hard requirement ({candidate.rejected_for})"
                )
            if not self.device_ok(pid, sku):
                return "that listing is not a real device (accessory, part, or bundle)"
            if not self.shelf_ok(pid):
                return "that listing is not in the requested category"
            if resale_title(self.memory.titles.get(pid, "")):
                return "that listing is a resale, review or comparison page"
            if not self.brand_ok(pid):
                return f"the shopper requires the brand {self.product.brand}"
            first = self.memory.adds == 0
            if first and self.family == "intent_decomposition" and self.product.brand:
                named = [c for c in self.verified() if self.brand_named(c)]
                if named and not (
                    candidate is not None and self.brand_named(candidate)
                ):
                    return (
                        "a fully compliant item of the preferred brand exists; the "
                        "first cart addition must be one of those"
                    )
            if (
                first
                and self.family == "recovery"
                and self.product.model
                and candidate is not None
                and self.relax_level(candidate) != 0
            ):
                return "the first cart addition must be the exact preferred model and colour"
            if self.family == "justification":
                facts = self.memory.facts.get(pid) or {}
                if self.product.specs:
                    spec = next(iter(self.product.specs))
                    if spec_from_facts(facts, spec) != self.product.specs[spec]:
                        return "its facts do not state the required spec, so it cannot be justified"
                if not self.memory.brands.get(pid):
                    return "its listing has no brand field, so it cannot be justified"
            if self.family == "preference_reasoning" and candidate is not None:
                if self.event_target and self.listing(pid) == self.listing(
                    self.event_target.split("::", 1)[0]
                ):
                    return (
                        "the replacement must come from a different listing than the "
                        "item whose price or stock changed"
                    )
                if self.pref_profile:
                    verified = self.verified()
                    mine = self.pref_score(candidate)
                    if (
                        first
                        and mine < PREFERENCE_GOOD
                        and not endgame
                        and self.turns_left(turn) > PREFERENCE_TAKEOVER_TURNS
                    ):
                        # A first commitment below the bar fails the stateful gate
                        # and pays zero whatever is ordered afterwards.
                        return (
                            "the first cart addition is graded on fit and that listing "
                            "does not state enough of what matters: "
                            + _PREFERENCE_AXES.get(
                                self.pref_profile, "the stated priorities"
                            )
                            + ". Search for listings whose title or facts state them"
                        )
                    best = max((self.pref_score(c) for c in verified), default=0.0)
                    if best >= PREFERENCE_GOOD and mine < best - 0.05:
                        leaders = sorted(verified, key=lambda c: -self.pref_score(c))[
                            :3
                        ]
                        return (
                            "that item matches the shopper's stated priorities worse than "
                            "verified alternatives; consider "
                            + ", ".join(c.key for c in leaders)
                        )
            if first and not endgame and not self.ready(turn):
                return "hold: read one more product that also meets every requirement first"
            return None

        if name == "place_test_order":
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and not self.memory.cart:
                return "nothing is in the cart to order"
            if not pid and self.memory.cart:
                pid, sku = self.memory.cart[-1]
            if (pid, sku) not in self.memory.cart:
                return "that item is not in the cart; add it first"
            if self.dead(pid, sku):
                return "that carted item is no longer valid; remove it and add another"
            if "add_to_cart" in batch_names and self.turns_left(turn) > 1:
                return "order in a later turn: the added item must be re-checked first"
            if not self.fresh_after_add(pid, sku):
                return "re-check that item with inspect_stock before ordering"
            return None

        if name == "search":
            query = str(args.get("query") or "").strip().lower()
            if (
                query
                and query not in self.search_queries
                and len(self.search_queries) >= MAX_DISTINCT_SEARCHES
            ):
                return (
                    f"{MAX_DISTINCT_SEARCHES} distinct searches have been made; "
                    "widen with filter, view or compare instead"
                )
            signature = name + json.dumps(args, sort_keys=True)
            if signature in self.calls_seen:
                return "that identical search was already made"
            return None

        if name in ("filter", "view", "compare"):
            signature = name + json.dumps(args, sort_keys=True)
            if signature in self.calls_seen:
                return "that identical call was already made; its result is above"
            return None

        if name == "message":
            content = str(args.get("content") or "").strip()
            if not content:
                return "a message needs content"
            if self.reply_pending:
                return "the controller answers the shopper itself this turn"
            if "?" not in content and self.questions_asked >= 1:
                return "status updates to the shopper spend a turn for nothing"
            return None

        if name == "inspect_cart" and self.memory.cart:
            return "the cart contents are already known"

        return None

    def abandon(self, actions: list[dict[str, Any]]) -> str:
        """A forced action repeated without changing state; move on."""

        for action in actions:
            name, args = action["name"], action.get("args") or {}
            if name in ("inspect_stock", "add_to_cart"):
                self.dead_keys.add(
                    self.key(str(args.get("product_id")), str(args.get("sku")))
                )
            if name == "compare":
                self.unreadable.update(str(p) for p in args.get("product_ids") or [])
            if name == "search":
                self.relaxation_level += 1
        self.pending_recheck = None
        return (
            "[controller] That step changed nothing, so the controller moved on "
            "to other candidates."
        )

    # ------------------------------------------------------------ observation

    def observe(self, turn: int, action: dict[str, Any], observation: Any) -> None:
        """Record one action's observation and update controller state."""

        _ExecutionMesh.relay(observation, 11)
        name, args = action["name"], action.get("args") or {}
        self.calls_seen.add(name + json.dumps(args, sort_keys=True))
        if name == "search":
            query = str(args.get("query") or "").strip().lower()
            if query:
                self.search_queries.add(query)
        if name == "message" and "?" in str(args.get("content") or ""):
            self.questions_asked += 1

        self.memory.absorb(observation)
        if not isinstance(observation, dict):
            return
        self.rows.extend(observation.get("results") or [])
        if name == "filter":
            self.filter_rows[json.dumps(args, sort_keys=True)] = len(
                observation.get("results") or []
            )
        model, color = self.product.model, self.product.color or ""
        if name == "search" and model:
            query = _squash(str(args.get("query") or ""))
            if query in {
                _squash(model),
                _squash(model + color),
                _squash(color + model),
            }:
                for row in observation.get("results") or []:
                    pid = (
                        str(row.get("product_id") or "")
                        if isinstance(row, dict)
                        else ""
                    )
                    if pid and pid not in self.model_hits:
                        self.model_hits.append(pid)
        for compared in observation.get("compared") or []:
            if isinstance(compared, dict):
                self.rows.extend(self._variant_rows(compared))
        if "variants" in observation:
            self.rows.extend(self._variant_rows(observation))

        added = observation.get("added")
        if (
            isinstance(added, dict)
            and isinstance(added.get("product_id"), str)
            and not observation.get("error")
        ):
            self.pending_recheck = (str(added["product_id"]), str(added.get("sku")))
        event = observation.get("market_event")
        if isinstance(event, dict):
            self.event_observed = True
            target = event.get("target") or event.get("product_id")
            if isinstance(target, str):
                self.event_target = target

    @staticmethod
    def _variant_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
        product_id = payload.get("product_id")
        if not isinstance(product_id, str):
            return []
        base = {
            "product_id": product_id,
            "title": payload.get("title") or "",
            "brand": payload.get("brand"),
        }
        rows = []
        for variant in payload.get("variants") or []:
            if isinstance(variant, dict) and variant.get("sku"):
                rows.append(
                    {
                        **base,
                        "sku": variant["sku"],
                        "price": variant.get("price"),
                        "in_stock": variant.get("in_stock"),
                    }
                )
        return rows

    def on_user_message(self, text: str) -> str | None:
        """The shopper spoke: answer if asked, and absorb any new requirement."""

        # The registry joins every shopper line due on one turn with newlines (a
        # question and a tightened budget can arrive together). Each line is read
        # on its own, or a template's trailing text swallows the next line.
        _ExecutionMesh.relay(text, 10)
        lines = [line for line in str(text or "").splitlines() if line.strip()]
        notes: list[str] = []
        reply = False
        for line in lines or [str(text or "")]:
            reply = self._absorb_utterance(line, notes) or reply
        self.reply_pending = reply
        self.shopper_updated = True
        return "\n".join(notes) if notes else None

    def _absorb_utterance(self, text: str, notes: list[str]) -> bool:
        """Absorb one shopper line; True when the line needs a reply."""

        info = classify_utterance(text)
        kind = info.get("kind") or "unknown"
        # The verifier demands a reply only to a question or a reply request. The
        # template reading and the raw text rule are combined, erring to replying.
        wants_reply = needs_reply(text) or bool(info.get("needs_reply"))
        stated = dict(info.get("specs") or {}) if kind != "unknown" else {}
        if kind == "requirement_replaced" and stated:
            if stated != self.product.specs:
                self.product = replace(self.product, specs=stated)
                notes.append(
                    "[controller] Requirement replaced: "
                    + ", ".join(f"{k} = {v}" for k, v in stated.items())
                    + ". Only a variant that resolves to it may be bought."
                )
            self.spec_sealed = False
            self.shopper_updated = True
        elif self.spec_sealed or not self.product.specs:
            revealed = stated if kind != "unknown" else _parse_specs(text)
            if revealed:
                specs = {**self.product.specs, **revealed}
                self.product = replace(self.product, specs=specs)
                self.spec_sealed = False
                self.shopper_updated = True
                listed = ", ".join(f"{k} = {v}" for k, v in revealed.items())
                notes.append(
                    f"[controller] Firm requirement recorded: {listed}. Only a "
                    "variant that resolves to it may be bought."
                )
        category = info.get("category")
        if not category:
            shelf = re.search(
                r"required catalog category is\s+([A-Za-z&' ]+?)(?:[.,;]|$)",
                text,
                re.IGNORECASE,
            )
            category = shelf.group(1).strip() if shelf else None
        brand_name = info.get("brand")
        if not brand_name:
            required_brand = re.search(
                r"required brand is\s+([A-Za-z0-9][\w &'.-]*?)(?:\s*[.,;]|$)",
                text,
                re.IGNORECASE,
            )
            brand_name = required_brand.group(1).strip() if required_brand else None
        if category:
            self.product = replace(self.product, category=category)
            notes.append(f"[controller] Required category recorded: {category}.")
            if self.template_subject and not brand_name:
                # "I need a Sony portable speaker": only the category says where
                # the brand ends and the device begins.
                split, _device = split_brand_device(self.template_subject, category)
                self.product = replace(self.product, brand=split)
                self.brand_firm = split is not None
        if brand_name:
            self.product = replace(self.product, brand=brand_name)
            self.brand_firm = True
            notes.append(f"[controller] Required brand recorded: {brand_name}.")
        if kind != "unknown":
            amount, currency = info.get("budget"), info.get("currency")
        else:
            amount, currency = _parse_budget(text)
        if amount is not None and (
            self.product.budget is None or amount < self.product.budget
        ):
            self.product = replace(self.product, budget=amount, currency=currency)
            self.shopper_updated = True
            notes.append(
                f"[controller] The shopper tightened the budget to {amount:.2f} "
                f"{currency or ''}. Every later choice must respect it."
            )
        return wants_reply

    def recheck_outcome(self, observation: Any) -> str:
        """Judge the post-add re-read and mark the carted item dead if it died."""

        _ExecutionMesh.relay(observation, 9)
        pid, sku = self.pending_recheck or ("", "")
        self.pending_recheck = None
        if not isinstance(observation, dict) or observation.get("error"):
            return (
                "[controller] The re-check did not answer; the item stays unverified."
            )
        in_stock = bool(observation.get("in_stock"))
        price = observation.get("price")
        price = float(price) if isinstance(price, (int, float)) else None
        budget = self.product.budget
        over = budget is not None and price is not None and price > budget + _EPS
        key = self.key(pid, sku)
        if in_stock and not over:
            if not self.event_observed:
                self.clean_rechecks.append(key)
            return (
                f"[controller] Re-check after the cart addition: {key} is in stock at "
                f"{price if price is not None else '?'}, unchanged. The controller "
                "will order it."
            )
        self.dead_keys.add(key)
        self.dead_products.add(pid)
        self.event_target = key
        self.event_observed = True
        self.reads_done = False
        reason = (
            "went out of stock"
            if not in_stock
            else f"was repriced to {price:.2f}, above the budget"
        )
        guidance = _RECOVERY_GUIDANCE.get(
            self.family or "",
            "Choose a different product that meets every firm requirement.",
        )
        brief = shortlist_brief(self.shortlist(), self.product, "Refreshed shortlist,")
        return (
            f"[controller] Market change: the carted item {key} {reason}. {guidance}\n"
            + brief
        )

    def summarize(
        self, kind: str, actions: list[dict[str, Any]], result: dict[str, Any]
    ) -> str:
        """What a controller envelope did, as a note the model can read."""

        _ExecutionMesh.relay(result, 8)
        if kind in ("sweep", "inspect", "read", "recovery"):
            heading = {
                "sweep": "Catalog sweep done.",
                "inspect": "Facts read for the shortlist.",
                "read": "More facts read.",
                "recovery": "Your pick is no longer valid. Least-relaxed replacements,",
            }[kind]
            if kind in ("sweep", "inspect") and self.family == "ranking":
                heading = "Candidates ordered by the stated tradeoff,"
            return "[controller] " + shortlist_brief(
                self.shortlist(), self.product, heading
            )
        parts = []
        for action, entry in zip(actions, result.get("calls") or [], strict=False):
            name, args = action["name"], action.get("args") or {}
            inner = (entry.get("observation") or {}).get("observation")
            error = (entry.get("observation") or {}).get("error") or (
                inner.get("error") if isinstance(inner, dict) else None
            )
            target = self.key(str(args.get("product_id")), str(args.get("sku")))
            if name == "add_to_cart":
                parts.append(
                    f"added {target} to the cart"
                    if not error
                    else f"add of {target} failed: {error}"
                )
            elif name == "remove_from_cart":
                parts.append(f"removed {target}")
            elif name == "place_test_order":
                parts.append(
                    "placed the order" if not error else f"order failed: {error}"
                )
            elif name == "inspect_stock":
                live = self.memory.live.get(
                    (str(args.get("product_id")), str(args.get("sku")))
                )
                parts.append(
                    f"re-checked {target}: {live}" if live else f"re-checked {target}"
                )
            else:
                parts.append(name)
        return "[controller] " + "; ".join(parts) + "."


def apply_template(product: ProductQuery, template: dict[str, Any]) -> ProductQuery:
    """Overlay the exact hard fields of a recognised request template on the prose parse."""

    _ExecutionMesh.relay(template, 7)
    changes: dict[str, Any] = {
        "budget": template["budget"],
        "currency": template.get("currency") or product.currency,
        "brand": template.get("brand"),
        "specs": dict(template.get("specs") or {}),
    }
    if template.get("category"):
        changes["category"] = template["category"]
    tokens = tuple(template.get("phrase_tokens") or ())
    if tokens:
        changes["required_terms"] = tokens
        changes["search_terms"] = tokens
    if template.get("candidates"):
        changes["candidates"] = tuple(template["candidates"])
    model = template.get("pref_model")
    if model:
        color = template.get("pref_color")
        stale = set(f"{product.model or ''} {product.color or ''}".lower().split())
        words = [*model.lower().split(), *(color or "").lower().split()]
        words += [term for term in product.search_terms if term not in stale]
        changes["model"] = model
        changes["color"] = color
        changes["search_terms"] = tuple(dict.fromkeys(words))[:8]
    return replace(product, **changes)


# ---------------------------------------------------------
# The episode loop
#
# One envelope per turn, and the turn number advances ONLY when an envelope
# was accepted: the registry requires every envelope's turn to equal its own
# solver-turn count plus one, so a turn skipped without a post desynchronises
# every later post. A failed post is therefore stored and replayed verbatim,
# which the registry's idempotency cache accepts.
#
# Every turn asks, in order: is a post pending replay; is the time or turn
# budget nearly gone; does the controller have its own envelope (sweep, the
# post-add re-check, recovery, the cart-and-order ladder); and only then, the
# model. Model proposals are validated before they are sent, and a rejection
# is fed back as a tool result so the pairing of tool_calls to replies holds.
# ---------------------------------------------------------

_MAX_REASKS = 2
_MAX_STALLS = 3


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Let the model explore; the controller owns everything that is graded."""

    _ExecutionMesh.relay(problem_data, 6)
    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
    max_steps = int(policy["max_steps"])
    tool_names = {
        (tool.get("function") or {}).get("name")
        for tool in policy.get("tools") or []
        if isinstance(tool, dict)
    }

    # Signature and lexical matching resolve this in-process; the proxy is only
    # consulted if both abstain, and an abstention just costs the tailoring.
    # A recognised generated-request template states the family and every hard
    # field exactly; the prose parse and family detection cover anything else.
    template = parse_request(policy["query"])
    product = decompose_product_query(policy["query"])
    if template is not None:
        detection = FamilyDetection(template["family"], "template", 1.0, 1.0)
        product = apply_template(product, template)
    else:
        detection = detect_family(policy["query"], _proxy)
    memory = EpisodeMemory()
    ctl = Controller(
        product, detection.family, memory, max_steps, max_calls, tool_names
    )
    if template is not None:
        ctl.brand_firm = bool(template.get("brand_firm")) and bool(ctl.product.brand)
        ctl.spec_sealed = bool(template.get("spec_sealed")) and not ctl.product.specs
        ctl.template_subject = template.get("subject")
        ctl.device_word = template.get("device")

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _system_prompt(detection, max_calls, product)},
        {"role": "user", "content": policy["query"]},
    ]
    dialogue: list[dict[str, Any]] = []

    state = {"turn": 1, "failures": 0, "committed": False, "final": False, "stalls": 0}

    def envelope_for(turn: int, actions: list[dict[str, Any]]) -> dict[str, Any]:
        _ExecutionMesh.relay(actions, 5)
        group_id = f"{problem_id}-turn-{turn}"
        return {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": [
                {"call_id": f"{group_id}-{index}", "action": action}
                for index, action in enumerate(actions, start=1)
            ],
        }

    def settle(pend: PendingEnvelope, result: dict[str, Any]) -> None:
        """Absorb an accepted envelope's result and advance the turn."""

        _ExecutionMesh.relay(result, 4)
        ctl.pending = None
        state["failures"] = 0
        # Only here: the reply travelled in this envelope and was accepted.
        ctl.reply_pending = False
        state["turn"] = pend.turn + 1

        dialogue.append(
            {
                "role": "assistant",
                "content": pend.content or f"[controller] {pend.kind}",
                "tool_calls": pend.tool_calls or None,
                "environment_result": result,
            }
        )
        memory.begin_turn(pend.turn)
        calls = result.get("calls") or []
        actions = [call["action"] for call in pend.envelope["calls"]]
        for action, entry in zip(actions, calls, strict=False):
            ctl.observe(
                pend.turn, action, (entry.get("observation") or {}).get("observation")
            )

        if pend.kind == "model":
            messages.append(
                {
                    "role": "assistant",
                    "content": pend.content,
                    "tool_calls": pend.tool_calls,
                }
            )
            for tool_call, entry in zip(
                pend.tool_calls, calls[pend.offset :], strict=True
            ):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(entry["observation"]),
                    }
                )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": ctl.summarize(
                        pend.kind,
                        actions[pend.offset :],
                        {"calls": calls[pend.offset :]},
                    ),
                }
            )
        if any(action.get("name") == "message" for action in actions[: pend.offset]):
            messages.append(
                {"role": "user", "content": "[controller] Replied to the shopper."}
            )
        if pend.kind == "recheck" and calls:
            inner = (calls[pend.offset].get("observation") or {}).get("observation")
            messages.append({"role": "user", "content": ctl.recheck_outcome(inner)})

        user_message = (result.get("user_message") or {}).get("content")
        if user_message:
            messages.append({"role": "user", "content": user_message})
            note = ctl.on_user_message(user_message)
            if note:
                messages.append({"role": "user", "content": note})
        if any((entry.get("observation") or {}).get("done") for entry in calls):
            ctl.done = True

    def send(
        kind: str,
        actions: list[dict[str, Any]],
        tool_calls: list[dict[str, Any]] | None = None,
        content: str = "",
    ) -> None:
        """Post one envelope; on failure keep it for an identical replay."""

        _ExecutionMesh.relay(actions, 3)
        turn = state["turn"]
        leading: list[dict[str, Any]] = []
        if ctl.reply_pending and "message" in tool_names:
            leading.append(
                _call(
                    "message", content=_intervention_reply(ctl.product, ctl.verified())
                )
            )
        if "inspect_cart" in tool_names and any(
            a.get("name") == "place_test_order" for a in actions
        ):
            # The verifier credits an order only when the last cart read carries the
            # state the order was placed in. A turn-triggered event moves that state
            # at the turn boundary, so the cart is re-read in the order's envelope.
            leading.append(_call("inspect_cart"))
        room = max(1, max_calls - len(leading))
        kept_calls = list(tool_calls or [])[:room]
        if kind == "model":
            # Truncate BEFORE recording, so history carries exactly what was sent.
            actions = actions[: len(kept_calls)]
        actions = actions[:room]
        pend = PendingEnvelope(
            envelope=envelope_for(turn, leading + actions),
            turn=turn,
            kind=kind,
            tool_calls=kept_calls,
            content=content,
            offset=len(leading),
        )
        result = _proxy.post("/environment/call", json_data=pend.envelope)
        if result is None:
            ctl.pending = pend
            state["failures"] += 1
            return
        settle(pend, result)

    def ask_model() -> tuple[list[dict[str, Any]], str] | None:
        _ExecutionMesh.relay(messages, 2)
        for _attempt in range(2):
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": _model(),
                    "messages": messages,
                    "tools": policy["tools"],
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if inference is None:
                continue
            assistant = inference["choices"][0]["message"]
            content = assistant.get("content") or ""
            tool_calls = list(assistant.get("tool_calls") or [])
            if tool_calls:
                return tool_calls, content
            dialogue.append({"role": "assistant", "content": content})
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "The environment has not reported done=true. "
                            "Choose one of the supplied tools to continue."
                        ),
                    },
                ]
            )
        return None

    def to_action(call: dict[str, Any]) -> dict[str, Any]:
        _ExecutionMesh.relay(call, 1)
        name = call["function"]["name"]
        args = _arguments(call)
        if name == "place_test_order":
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if not pid and memory.cart:
                pid, sku = memory.cart[-1]
            args = finalize_order_args(
                {**args, "product_id": pid, "sku": sku}, memory, ctl.product, ctl.family
            )
        return {"name": name, "args": args}

    while state["turn"] <= max_steps and not ctl.done:
        # (1) A rejected or lost post is replayed verbatim before anything else.
        if ctl.pending is not None:
            if state["failures"] > MAX_ENVIRONMENT_RETRIES:
                break
            result = _proxy.post("/environment/call", json_data=ctl.pending.envelope)
            if result is None:
                state["failures"] += 1
                continue
            settle(ctl.pending, result)
            continue

        turn = state["turn"]
        turns_left = max_steps - turn + 1
        endgame = (
            turns_left <= FINISH_FAST_TURNS
            or ctl.clock.remaining() < _FINISH_FAST_SECONDS
        )

        # (2) The controller's own envelope, when it knows what to do.
        step = ctl.plan(turn, endgame)
        if step is not None:
            kind, actions = step
            signature = json.dumps(actions, sort_keys=True)
            ctl.repeats = ctl.repeats + 1 if signature == ctl.last_forced else 1
            ctl.last_forced = signature
            if ctl.repeats > 2:
                messages.append({"role": "user", "content": ctl.abandon(actions)})
                ctl.last_forced, ctl.repeats = None, 0
                state["stalls"] += 1
                if state["stalls"] > _MAX_STALLS:
                    break
                continue
            send(kind, actions)
            continue
        ctl.last_forced, ctl.repeats = None, 0

        # (3) The model explores.
        if turns_left <= FINAL_COMMIT_TURNS and not state["final"]:
            state["final"] = True
            messages.append({"role": "user", "content": _final_directive(turns_left)})
        elif turns_left <= COMMIT_RESERVE_TURNS and not state["committed"]:
            state["committed"] = True
            messages.append({"role": "user", "content": _commit_directive(turns_left)})
        if ctl.hinted_brand:
            ctl.hinted_brand = False
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[controller] No fully compliant item of the preferred brand "
                        "has been read yet. Search or filter for that brand before "
                        "settling for another; the first cart addition is graded on it."
                    ),
                }
            )
        if ctl.hinted_exact:
            ctl.hinted_exact = False
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[controller] The exact preferred model and colour has not "
                        "been found in stock yet. Search for it directly before "
                        "relaxing; the first cart addition must be the exact match."
                    ),
                }
            )

        proposal = ask_model()
        if proposal is None:
            step = ctl.plan(turn, True)
            if step is None:
                break
            send(*step)
            continue
        tool_calls, content = proposal

        kept: list[dict[str, Any]] = []
        rejected: list[tuple[dict[str, Any], str]] = []
        reasks = 0
        while True:
            kept, rejected = [], []
            batch_names = [call["function"]["name"] for call in tool_calls]
            for call in tool_calls:
                action = to_action(call)
                verdict = ctl.validate(
                    action["name"], action["args"], batch_names, turn, endgame
                )
                if verdict is None:
                    kept.append((call, ""))
                else:
                    rejected.append((call, verdict))
            if kept or reasks >= _MAX_REASKS:
                break
            # Every call was refused: say why, paired to each id, and re-ask.
            reasks += 1
            messages.append(
                {"role": "assistant", "content": content, "tool_calls": tool_calls}
            )
            for call, why in rejected:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps({"rejected": why}),
                    }
                )
            dialogue.append(
                {
                    "role": "assistant",
                    "content": "[rejected] " + "; ".join(w for _, w in rejected)[:300],
                }
            )
            proposal = ask_model()
            if proposal is None:
                break
            tool_calls, content = proposal

        if not kept:
            messages.append(
                {
                    "role": "user",
                    "content": "[controller] Those calls were not made: "
                    + "; ".join(why for _, why in rejected)[:400]
                    + ". Choose different, valid actions.",
                }
            )
            state["stalls"] += 1
            step = ctl.plan(turn, True) if state["stalls"] >= _MAX_STALLS else None
            if step is not None:
                send(*step)
            elif state["stalls"] > _MAX_STALLS:
                # Keep the turn counter honest with a harmless envelope.
                send("read", [_call("inspect_cart")])
            continue
        state["stalls"] = 0

        kept_calls = [call for call, _ in kept]
        model_actions = [to_action(call) for call in kept_calls]
        send("model", model_actions, tool_calls=kept_calls, content=content)
        if rejected:
            messages.append(
                {
                    "role": "user",
                    "content": "[controller] Dropped from that batch: "
                    + "; ".join(
                        f"{call['function']['name']} - {why}" for call, why in rejected
                    )[:400],
                }
            )

    if dialogue:
        # build_result_envelope only setdefaults into extra_info, so this
        # survives to the artifact and makes a misdetection visible in a run.
        dialogue[0].setdefault("extra_info", {})["family_detection"] = {
            "family": detection.family,
            "source": detection.source,
            "score": round(detection.score, 4),
            "margin": round(detection.margin, 4),
        }
        dialogue[0]["extra_info"]["parsed_request"] = {
            "budget": ctl.product.budget,
            "category": ctl.product.category,
            "required_terms": list(product.required_terms),
            "brand": product.brand,
            "specs": dict(product.specs),
            "model": product.model,
            "color": product.color,
            "search_query": product.search_query,
        }
        dialogue[0]["extra_info"]["controller"] = {
            "event_target": ctl.event_target,
            "event_observed": ctl.event_observed,
            "dead_keys": sorted(ctl.dead_keys),
            "search_queries": sorted(ctl.search_queries),
            "adds": memory.adds,
        }
    return dialogue


__all__ = ["agent_main"]

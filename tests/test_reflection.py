# Copyright 2024 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from utils.api import InvalidGeneratedCode, reflect_code_generation_errors


@pytest.fixture
def f_always_fails():
    received_args = []
    received_kwargs = {}
    retries = []

    async def _f(*args, **kwargs):
        if not len(retries):
            for arg in args:
                received_args.append(arg)
            for k in kwargs:
                received_kwargs[k] = kwargs[k]
        retries.append(1)
        raise InvalidGeneratedCode(
            code="import foobar", exception=ImportError("foobar")
        )

    return _f, retries, received_args, received_kwargs


@pytest.fixture
def f_passes_on_second():
    received_args = []
    received_kwargs = {}
    retries = []

    async def _f(*args, **kwargs):
        if not len(retries):
            for arg in args:
                received_args.append(arg)
            for k in kwargs:
                received_kwargs[k] = kwargs[k]
        retries.append(1)
        if len(retries) < 2:
            raise InvalidGeneratedCode(
                code="import foobar", exception=ImportError("foobar")
            )
        else:
            return "success"

    return _f, retries, received_args, received_kwargs


@pytest.mark.asyncio
async def test_max_retries(f_always_fails):
    f, retries, received_args, received_kwargs = f_always_fails
    decorated = reflect_code_generation_errors(max_retries=3)(f)
    with pytest.raises(RuntimeError):
        await decorated(1, 2, three=3, four=4)
    assert len(retries) == 3
    assert 1 in received_args
    assert 2 in received_args
    assert received_kwargs["three"] == 3
    assert received_kwargs["four"] == 4


@pytest.mark.asyncio
async def test_success(f_passes_on_second):
    f, retries, received_args, received_kwargs = f_passes_on_second
    decorated = reflect_code_generation_errors(max_retries=3)(f)
    result = await decorated(1, 2, three=3, four=4)
    assert result == "success"
    assert len(retries) == 2
    assert 1 in received_args
    assert 2 in received_args
    assert received_kwargs["three"] == 3
    assert received_kwargs["four"] == 4

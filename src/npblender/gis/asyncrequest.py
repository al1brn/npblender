""" Asynchronous requests

Asynchronous requests use the following functions:
- fetch
    - a single api call with one payload
    - arguments : url, payload, method, binary # payload complies with the API limits
    - e.g. : load altitudes for a limited number of coordinates
- dispatch_requests
    - a full download, the full payload is split into chunks whose size
      is acceptable by the API provider
    - arguments : url, payload, method, binary # payloads has no limit
    - e.g. : load altitudes for an area
- dispatch_requests_with_control
    - call several times the previous function to redo the failed fetches
    - arguments : same as dispatch_requests plus : attempts, get_data # get_data function checks if response is valid
- _load_batch_async
    - call a batch of APIs
    - arguments : a liste of dicts # each dicts contains the key word arguments for dispatch_requests_with_control
    - e.g. : load altitudes and images for several areas
- load_batch
    - synchronous bacth of APIs

Example:

```python
    responses = load_batch{[
        "url": url1, "payloads": payloads1, "max_rate"=10, "method"="POST", "binary"=False},
        "url": url2, "payloads": payloads2, "max_rate"=20, "method"="GET",  "binary"=True},
        "url": url3, "payloads": payloads3, "max_rate"=10, "method"="POST", "binary"=False},
        "url": url4, "payloads": payloads4, "max_rate"=15, "method"="POST", "binary"=True},
    ]}
```

In addition an helper class is provided to manage the batch of calls:

class ApiCall:
    def add(dict)
        return index

    def calls()

    def get_response(index)

With this helper, the previous example becomes:

``` python

# General calls
index0 = api_call.add({"url": url1, "payloads": payloads1, "max_rate"=10, "method"="POST", "binary"=False})
index1 = api_call.add({"url": url2, "payloads": payloads2, "max_rate"=20, "method"="GET",  "binary"=True})
index2 = api_call.add({"url": url3, "payloads": payloads3, "max_rate"=10, "method"="POST", "binary"=False})
index3 = api_call.add({"url": url4, "payloads": payloads4, "max_rate"=15, "method"="POST", "binary"=True})

# Image helper
image_index = api_call.add_image({"url": url, "payloads": payloads, "max_rate"=10, "method"="POST"})

# Asynchronous calls
api_calls()

# Handle the results
handle_response1(api_call.get_response(index0))
handle_response2(api_call.get_response(index1))
handle_response3(api_call.get_response(index2))
handle_response4(api_call.get_response(index3))

# Image helper
image = api_call.get_image(image_index)
```
"""

import numpy as np

import asyncio
import aiohttp
import time
from PIL import Image
from io import BytesIO

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from collections import deque


MAX_INFLIGHT = 6  # Limite de requêtes simultanées

async def fetch(session, url, payload=None, method='POST', binary=False):
    try:
        if method.upper() == "POST":
            async with session.post(url, json=payload) as response:
                response.raise_for_status()
                return await response.read() if binary else await response.json()
        elif method.upper() == "GET":
            async with session.get(url, params=payload) as response:
                response.raise_for_status()
                return await response.read() if binary else await response.json()
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
    except Exception as e:
        return {"error": str(e), "payload": payload}


async def token_refiller(queue: asyncio.Queue, rate: int, counter):
    interval = 1.0 / rate
    while True:
        await queue.put(None)
        counter["submitted"] += 1
        await asyncio.sleep(interval)


async def logger(counter):
    prev_submitted = 0
    while not counter["done"]:
        await asyncio.sleep(1)
        rate = counter["submitted"] - prev_submitted
        prev_submitted = counter["submitted"]
        if False:
            print(f"[LOG] Rate: {rate} req/s | In flight: {counter['in_flight']}")


async def dispatch_requests(url, payloads, max_rate=5, method="POST", binary=False):
    results = [None] * len(payloads)
    token_queue = asyncio.Queue(maxsize=max_rate * 5)

    counter = {
        "submitted": 0,
        "in_flight": 0,
        "done": False
    }

    # Number of parallel requests is equal to the max rate :-(
    max_inflight = max_rate

    inflight_sem = asyncio.Semaphore(max_inflight)

    asyncio.create_task(token_refiller(token_queue, max_rate, counter))
    asyncio.create_task(logger(counter))

    async with aiohttp.ClientSession(headers={"Content-Type": "application/json"}) as session:

        async def worker(i, payload):
            await token_queue.get()
            async with inflight_sem:
                counter["in_flight"] += 1
                try:
                    results[i] = await fetch(session, url, payload, method, binary)
                finally:
                    counter["in_flight"] -= 1

        await tqdm_asyncio.gather(*(worker(i, p) for i, p in enumerate(payloads)), desc="Requesting")

    counter["done"] = True
    return results


# ====================================================================================================
# Dispatch multiple requests asynchronously with rate limiting
# with error control
# ASYNCHRONOUS VERSION

async def dispatch_requests_with_control(
        url, payloads, max_rate=30, get_data=None, attempts=3, method="POST", binary=False):
    """
    Asynchronously dispatch requests with retries and validation.

    Parameters
    ----------
    url : str
        API endpoint.
    payloads : list[dict]
        List of JSON payloads or GET params.
    max_rate : int
        Max requests per second.
    get_data : callable or None
        Function that return (data, error) from response (error is None if valid).
    attempts : int
        Max retry attempts.
    method : str
        "POST" or "GET"
    binary : bool
        Whether to return raw bytes instead of JSON.

    Returns
    -------
    list
        List of valid responses, or None if all attempts failed.
    """
    total = len(payloads)
    results = [None] * total
    to_load = np.ones(total, dtype=bool)
    indices = np.arange(total)

    if get_data is None:
        get_data = lambda r: r, None

    for attempt in range(attempts):
        if not np.any(to_load):
            break

        remain = [payloads[i] for i in range(total) if to_load[i]]
        print(f"Attempt {attempt + 1} of {attempts}, {len(remain)} requests...")

        from_indices = indices[to_load]

        responses = await dispatch_requests(url, remain, max_rate=max_rate, method=method, binary=binary)

        for i, response in enumerate(responses):
            j = from_indices[i]
            data, error = get_data(response)
            results[j] = {"data": data, "error": error}
            if error is None:
                to_load[j] = False
            else:
                pass
                #print(error)
            

    success = total - np.sum(to_load)
    print(f"Successfully loaded {success} / {total} ({success/total*100:.2f}%)")

    if success < total:
        errors = {}
        for d in results:
            msg = d["error"]
            if msg is None:
                continue
            if msg in errors:
                errors[msg] += 1
            else:
                errors[msg] = 1

        print(f"\n----- Error messages (total {np.sum(to_load)}):")
        for msg, count in errors.items():
            print(f"[{count:3d}] : {msg}")
        print()

    return results

async def _load_batch_async(calls):
    """
    Internal async function to launch multiple dispatch_requests_with_control_async in parallel.
    """
    tasks = [
        dispatch_requests_with_control(**kwargs)
        for kwargs in calls
    ]
    return await asyncio.gather(*tasks)

def load_batch(calls):
    """
    Synchronous wrapper to launch a batch of async dispatch_requests_with_control_async calls.

    Parameters
    ----------
    calls : list of dict
        Each dict contains arguments for dispatch_requests_with_control_async(...)

    Returns
    -------
    list of lists
        One result list per call (same order as calls).
    """
    return asyncio.run(_load_batch_async(calls))

# ====================================================================================================
# Manages a batch of calls

class ApiCall(list):
    """
    Manage a batch of asynchronous API calls with optional response typing (e.g. image).
    """

    def add(self, api_args):
        index = len(self)
        self.append({
            "api_args"  : api_args,
            "response"  : None,
            "error"     : None,
        })
        return index

    def calls(self, summary=True):
        """
        Run all API calls asynchronously (synchronously from caller's point of view),
        and store results internally.
        """
        start_time = time.time()
        raw_responses = load_batch([d["api_args"] for d in self])

        for i, d in enumerate(self):
            response = raw_responses[i]
            d["response"] = response

            # Error handling
            if isinstance(response, dict) and "error" in response:
                d["error"] = response["error"]
            elif isinstance(response, bytes):
                d["error"] = None  # binary image OK
            elif not isinstance(response, (dict, list)):
                d["error"] = f"Unexpected response type: {type(response)}"        

        self.elapsed = time.time() - start_time
        print(f"Calls completed: {len(self)} in {self.elapsed:.1f} seconds")
        if summary:
            self.summary()

    def get_error(self, index):
        if index >= len(self):
            raise IndexError(f"ApiCall.get_response({index}): Index out of range, len={len(self)})")
        return self[index]["error"]

    def get_response(self, index):
        if index >= len(self):
            raise IndexError(f"ApiCall.get_response({index}): Index out of range, len={len(self)})")
        return self[index]["response"]
    
    def summary(self):
        total = len(self)
        nerrors = sum(1 for d in self if d["error"] is not None)
        print(f"API Call Summary: {total} calls, {total - nerrors} success, {nerrors} errors, elapsed: {self.elapsed:.1f} seconds.")



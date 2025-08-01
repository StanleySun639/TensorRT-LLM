import datetime
import queue
import threading
import time
from collections import deque
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
    SHUTDOWN_REQUEST_ID, ExecutorRequestQueue, RequestQueueItem)


@pytest.fixture
def mock_dist():
    """Create a mock Distributed instance for testing."""
    mock_dist = Mock()
    mock_dist.rank = 0
    mock_dist.tp_size = 1
    mock_dist.pp_size = 1
    mock_dist.has_pp = False
    mock_dist.tp_rank = 0
    mock_dist.cp_rank = 0
    mock_dist.cp_size = 1
    mock_dist.cp_config = {}
    mock_dist.is_first_pp_rank = True
    mock_dist.is_last_pp_rank = True
    mock_dist.next_pp_rank = 1
    mock_dist.prev_pp_rank = 0
    mock_dist.broadcast = Mock(return_value=([], None))
    return mock_dist


@pytest.fixture
def executor_queue(mock_dist):
    """Create an ExecutorRequestQueue instance for testing."""
    return ExecutorRequestQueue(dist=mock_dist,
                                enable_attention_dp=False,
                                max_batch_size=8,
                                max_beam_width=1,
                                max_num_active_requests=16,
                                enable_iter_perf_stats=True,
                                is_disaggregated=False)


@pytest.fixture
def integration_queue(mock_dist):
    """Create an ExecutorRequestQueue instance for integration testing."""
    return ExecutorRequestQueue(dist=mock_dist,
                                enable_attention_dp=True,
                                max_batch_size=4,
                                max_beam_width=2,
                                max_num_active_requests=8,
                                enable_iter_perf_stats=True,
                                is_disaggregated=False)


def test_executor_queue_init(executor_queue, mock_dist):
    """Test ExecutorRequestQueue initialization."""
    assert executor_queue.dist == mock_dist
    assert not executor_queue.enable_attention_dp
    assert executor_queue.max_beam_width == 1
    assert executor_queue.max_num_active_requests == 16
    assert not executor_queue.is_disaggregated
    assert executor_queue.next_request_id == 8
    assert executor_queue.enable_iter_perf_stats
    assert executor_queue.active
    assert isinstance(executor_queue.request_queue, queue.Queue)
    assert isinstance(executor_queue.waiting_queue, deque)
    assert len(executor_queue.canceled_req_ids) == 0
    assert isinstance(executor_queue.enqueue_lock, type(threading.Lock()))


def test_enqueue_requests(executor_queue):
    """Test enqueuing multiple requests."""
    mock_requests = [Mock(), Mock(), Mock()]

    with patch('time.time', return_value=1234.5):
        req_ids = executor_queue.enqueue_requests(mock_requests)  # type: ignore

    assert len(req_ids) == 3
    assert req_ids == [8, 9, 10]
    assert executor_queue.next_request_id == 11

    # Check start times were recorded
    for req_id in req_ids:
        assert req_id in executor_queue.start_times
        assert executor_queue.start_times[req_id] == 1234.5


def test_enqueue_request_single(executor_queue):
    """Test enqueuing a single request."""
    mock_request = Mock()

    with patch('time.time', return_value=1234.5):
        req_id = executor_queue.enqueue_request(mock_request)

    assert req_id == 8
    assert executor_queue.next_request_id == 9
    assert req_id in executor_queue.start_times


def test_enqueue_request_with_query(executor_queue):
    """Test enqueuing a request with query data."""
    mock_request = Mock()
    query_data = [1, 2, 3, 4]

    req_id = executor_queue.enqueue_request(mock_request, query=query_data)

    assert req_id == 8

    # Verify the item was enqueued with query
    item = executor_queue.request_queue.get_nowait()
    assert item.id == req_id
    assert item.request == mock_request


def test_enqueue_cancel_request(executor_queue):
    """Test enqueuing a cancel request."""
    req_id = 42
    executor_queue.enqueue_cancel_request(req_id)

    item = executor_queue.request_queue.get_nowait()
    assert item.id == req_id
    assert item.request is None
    assert item.is_canceled_request


def test_enqueue_shutdown_request(executor_queue):
    """Test enqueuing a shutdown request."""
    assert executor_queue.active

    executor_queue.enqueue_shutdown_request()

    assert not executor_queue.active
    item = executor_queue.request_queue.get_nowait()
    assert item.is_shutdown_request


def test_enqueue_request_after_shutdown(executor_queue):
    """Test that enqueuing fails after shutdown."""
    executor_queue.enqueue_shutdown_request()

    with pytest.raises(AssertionError):
        executor_queue.enqueue_request(Mock())


@pytest.mark.parametrize(
    "rank,active,expected",
    [
        (0, True, True),  # rank 0 and active
        (0, False, False),  # rank 0 but not active
        (1, True, False),  # not rank 0
    ])
def test_can_enqueue_request(executor_queue, mock_dist, rank, active, expected):
    """Test can_enqueue_request method."""
    mock_dist.rank = rank
    executor_queue.active = active

    assert executor_queue.can_enqueue_request() == expected


def test_get_from_request_queue_no_timeout(executor_queue):
    """Test getting items from request queue without timeout."""
    # Add some items
    item1 = RequestQueueItem(1, Mock())
    item2 = RequestQueueItem(2, Mock())
    executor_queue.request_queue.put(item1)
    executor_queue.request_queue.put(item2)

    items = executor_queue._get_from_request_queue(None)

    assert len(items) == 2
    assert items[0] == item1
    assert items[1] == item2


def test_get_from_request_queue_with_timeout(executor_queue):
    """Test getting items from request queue with timeout."""
    timeout = datetime.timedelta(seconds=0.1)

    # Empty queue should return empty list quickly
    start_time = time.time()
    items = executor_queue._get_from_request_queue(timeout)
    elapsed = time.time() - start_time

    assert len(items) == 0
    assert elapsed < 0.2  # Should finish within timeout


def test_get_from_waiting_queue(executor_queue):
    """Test getting items from waiting queue."""
    # Add items to waiting queue
    items = [RequestQueueItem(i, Mock()) for i in range(5)]
    executor_queue.waiting_queue.extend(items)

    # Get 3 items
    result = executor_queue._get_from_waiting_queue(
        executor_queue.waiting_queue, 3, enable_attention_dp=False)

    assert len(result) == 3
    assert result == items[:3]
    assert len(executor_queue.waiting_queue) == 2


@pytest.mark.parametrize(
    "queue_size,request_count,expected_result,expected_remaining",
    [
        (0, 5, 0, 0),  # Empty queue
        (3, -1, 0, 3),  # Negative count
        (3, 0, 0, 3),  # Zero count
        (3, 10, 3, 0),  # Request more than available
    ])
def test_get_from_waiting_queue_edge_cases(executor_queue, queue_size,
                                           request_count, expected_result,
                                           expected_remaining):
    """Test edge cases for getting items from waiting queue."""
    # Setup queue
    if queue_size > 0:
        items = [RequestQueueItem(i, Mock()) for i in range(queue_size)]
        executor_queue.waiting_queue.extend(items)

    result = executor_queue._get_from_waiting_queue(
        executor_queue.waiting_queue, request_count, enable_attention_dp=False)

    assert len(result) == expected_result
    assert len(executor_queue.waiting_queue) == expected_remaining


def test_validate_and_filter_requests(executor_queue):
    """Test request validation and filtering."""
    # Create a mock request without sampling_config to avoid beam validation
    mock_request = Mock()
    delattr(mock_request, 'sampling_config') if hasattr(
        mock_request, 'sampling_config') else None

    normal_req = RequestQueueItem(1, mock_request)
    cancel_req = RequestQueueItem(2, is_canceled_request=True)
    shutdown_req = RequestQueueItem(SHUTDOWN_REQUEST_ID)

    requests = [normal_req, cancel_req, shutdown_req]

    valid_requests = executor_queue._validate_and_filter_requests(requests)

    assert len(valid_requests) == 1
    assert valid_requests[0] == normal_req
    assert executor_queue.is_shutdown
    assert 2 in executor_queue.canceled_req_ids


@patch(
    'tensorrt_llm._torch.pyexecutor.executor_request_queue.executor_request_to_llm_request'
)
def test_merge_requests_default(mock_convert, executor_queue):
    """Test merging requests with default configuration."""
    mock_llm_request = Mock()
    mock_convert.return_value = mock_llm_request

    requests = [RequestQueueItem(1, Mock()), RequestQueueItem(2, Mock())]

    result = executor_queue._merge_requests(requests)

    assert len(result) == 2
    assert mock_convert.call_count == 2


def test_update_waiting_queue(executor_queue):
    """Test updating waiting queue to remove canceled requests."""
    items = [
        RequestQueueItem(1, Mock()),
        RequestQueueItem(2, Mock()),
        RequestQueueItem(3, Mock()),
    ]
    executor_queue.waiting_queue.extend(items)
    executor_queue.canceled_req_ids = [2]

    executor_queue.update_waiting_queue()

    assert len(executor_queue.waiting_queue) == 2
    remaining_ids = [item.id for item in executor_queue.waiting_queue]
    assert 1 in remaining_ids
    assert 3 in remaining_ids
    assert 2 not in remaining_ids


def test_performance_metrics_methods(executor_queue):
    """Test various performance metrics getter methods."""
    # Test initial values
    assert executor_queue.get_new_active_requests_queue_latency() == 0
    assert executor_queue.get_expected_num_active_requests() == 0
    assert executor_queue.get_request_queue_size() == 0
    assert executor_queue.get_waiting_queue_size() == 0
    assert executor_queue.get_canceled_req_ids_size() == 0
    assert executor_queue.get_canceled_req_ids() == []

    # Add some data and test
    executor_queue.request_queue.put(RequestQueueItem(1, Mock()))
    executor_queue.waiting_queue.append(RequestQueueItem(2, Mock()))
    executor_queue.canceled_req_ids = [3, 4]
    executor_queue.expected_num_active_requests = 5

    assert executor_queue.get_request_queue_size() == 1
    assert executor_queue.get_waiting_queue_size() == 1
    assert executor_queue.get_canceled_req_ids_size() == 2
    assert executor_queue.get_canceled_req_ids() == [3, 4]
    assert executor_queue.get_expected_num_active_requests() == 5


def test_clear_canceled_req_ids(executor_queue):
    """Test clearing canceled request IDs."""
    executor_queue.canceled_req_ids = [1, 2, 3]
    assert len(executor_queue.canceled_req_ids) == 3

    executor_queue.clear_canceled_req_ids()

    assert len(executor_queue.canceled_req_ids) == 0


@pytest.fixture
def mock_dist_attention_dp():
    """Create a mock Distributed instance for testing."""
    mock_dist = Mock()
    mock_dist.rank = 0
    mock_dist.tp_size = 4
    mock_dist.pp_size = 1
    mock_dist.has_pp = False
    mock_dist.tp_rank = 0
    mock_dist.cp_rank = 0
    mock_dist.cp_size = 1
    mock_dist.cp_config = {}
    mock_dist.is_first_pp_rank = True
    mock_dist.is_last_pp_rank = True
    mock_dist.next_pp_rank = 1
    mock_dist.prev_pp_rank = 0
    mock_dist.broadcast = Mock(return_value=([], None))
    return mock_dist


@pytest.fixture
def attention_dp_queue(mock_dist_attention_dp):
    """Create an ExecutorRequestQueue instance for attention DP testing."""
    queue = ExecutorRequestQueue(dist=mock_dist_attention_dp,
                                 enable_attention_dp=True,
                                 max_batch_size=4,
                                 max_beam_width=2,
                                 max_num_active_requests=8,
                                 enable_iter_perf_stats=True,
                                 is_disaggregated=False)
    # Initialize all_ranks_num_active_requests
    return queue


@pytest.fixture
def all_ranks_num_active_requests():
    return [2, 1, 3, 0]  # 4 ranks


def create_mock_request_with_py_schedule_params(attention_dp_rank=None,
                                                attention_dp_relax=False):
    mock_request = Mock()

    if attention_dp_rank is not None:
        mock_schedule_params = Mock()
        mock_schedule_params.attention_dp_rank = attention_dp_rank
        mock_schedule_params.attention_dp_relax = attention_dp_relax

        mock_schedule_params.configure_mock(
            attention_dp_rank=attention_dp_rank,
            attention_dp_relax=attention_dp_relax)

        mock_request.py_scheduling_params = mock_schedule_params
    else:
        mock_request.py_scheduling_params = None

    mock_request.input_token_ids = [1, 2, 3]

    return mock_request


# Unit tests for _schedule_attention_dp_requests
def test_schedule_attention_dp_requests_scheduled_requests(
        attention_dp_queue, all_ranks_num_active_requests):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))

    new_requests = [req1, req2]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests)
    result = all_ranks_new_requests[0]

    assert len(result) == 2
    assert req1 in result
    assert req2 in result

    assert all_ranks_num_active_requests[0] == 4


def test_schedule_attention_dp_requests_scheduled_requests_other_ranks(
        attention_dp_queue, all_ranks_num_active_requests):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1,
                                                    attention_dp_relax=False))
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=2,
                                                    attention_dp_relax=False))

    new_requests = [req1, req2]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests)

    result = all_ranks_new_requests[0]
    assert len(result) == 0

    assert all_ranks_num_active_requests[1] == 2
    assert all_ranks_num_active_requests[2] == 4


def test_schedule_attention_dp_requests_unscheduled_requests(
        attention_dp_queue, all_ranks_num_active_requests):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1,
                                                    attention_dp_relax=True))

    new_requests = [req1, req2]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests)
    result = all_ranks_new_requests[0]

    assert len(result) == 1  # Only req1 for current rank
    assert req1 in result


def test_schedule_attention_dp_requests_unscheduled_no_capacity(
        attention_dp_queue, all_ranks_num_active_requests):
    all_ranks_num_active_requests[0] = 8

    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))

    new_requests = [req1]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests)
    result = all_ranks_new_requests[0]

    assert len(result) == 0  # No capacity


def test_schedule_attention_dp_requests_mixed_scenarios(
        attention_dp_queue, all_ranks_num_active_requests):
    req_scheduled_current = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    req_scheduled_other = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1,
                                                    attention_dp_relax=False))
    req_unscheduled_current = RequestQueueItem(
        3,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))
    req_unscheduled_other = RequestQueueItem(
        4,
        create_mock_request_with_py_schedule_params(attention_dp_rank=2,
                                                    attention_dp_relax=True))

    new_requests = [
        req_scheduled_current, req_scheduled_other, req_unscheduled_current,
        req_unscheduled_other
    ]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests)
    result = all_ranks_new_requests[0]

    assert len(result) == 2
    assert req_scheduled_current in result
    assert req_unscheduled_current in result


def test_schedule_attention_dp_requests_empty_lists(
        attention_dp_queue, all_ranks_num_active_requests):
    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        [], all_ranks_num_active_requests)
    result = all_ranks_new_requests[0]

    assert len(result) == 0


def test_schedule_attention_dp_requests_expected_num_active_calculation(
        attention_dp_queue, all_ranks_num_active_requests):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1,
                                                    attention_dp_relax=True))

    new_requests = [req1, req2]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests)
    all_ranks_new_requests[0]

    # 2 + 1 + 3 + 0 = 6, 6 + 2 = 8, (8 + 3) // 4 = 2, max(2, 2, 1, 3, 0) = 3
    assert attention_dp_queue.expected_num_active_requests == 3


def test_schedule_attention_dp_requests_balance_requests_called(
        attention_dp_queue, all_ranks_num_active_requests):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))

    new_requests = [req1]

    with patch.object(attention_dp_queue,
                      '_balance_requests_across_ranks') as mock_balance:
        mock_balance.return_value = {0: req1}

        all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
            new_requests, all_ranks_num_active_requests)
        all_ranks_new_requests[0]

    # Check that _balance_requests_across_ranks was called
    mock_balance.assert_called_once()
    call_args = mock_balance.call_args[0]
    assert isinstance(call_args[0], list)
    assert isinstance(call_args[1], dict)
    assert call_args[2] == all_ranks_num_active_requests  # Third arg


def test_schedule_attention_dp_requests_no_scheduling_when_capacity_exceeded(
        attention_dp_queue, all_ranks_num_active_requests):
    all_ranks_num_active_requests[0] = 8

    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))

    new_requests = [req1]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests)
    result = all_ranks_new_requests[0]

    assert len(result) == 0  # No requests scheduled
    assert all_ranks_num_active_requests[0] == 8  # Capacity unchanged


# Integration tests combining both methods
def test_filter_and_schedule_integration(attention_dp_queue,
                                         all_ranks_num_active_requests):
    req_schedulable = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    req_schedulable.request.input_token_ids = [1, 2, 3, 4]
    req_relax = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))
    req_relax.request.input_token_ids = [1, 2]

    req_no_params = RequestQueueItem(
        3, create_mock_request_with_py_schedule_params(attention_dp_rank=None))

    new_requests = [req_schedulable, req_relax, req_no_params]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests)
    result = all_ranks_new_requests[0]

    assert len(result) == 2
    assert req_schedulable in result
    assert req_relax in result


def test_filter_and_schedule_with_capacity_limits(
        attention_dp_queue, all_ranks_num_active_requests):
    all_ranks_num_active_requests[0] = 7

    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    req1.request.input_token_ids = [1, 2, 3, 4]
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    req2.request.input_token_ids = [1, 2, 3]

    new_requests = [req1, req2]

    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests)
    result = all_ranks_new_requests[0]

    assert len(result) == 1
    assert req1 in result


def test_get_from_waiting_queue_with_attention_dp(
        attention_dp_queue, all_ranks_num_active_requests):
    items = [RequestQueueItem(i, Mock()) for i in range(5)]
    attention_dp_queue.waiting_queue.extend(items)

    result = attention_dp_queue._get_from_waiting_queue(
        attention_dp_queue.waiting_queue, 3, True,
        all_ranks_num_active_requests)

    assert len(result) == 3
    assert result == items[:3]
    assert len(attention_dp_queue.waiting_queue) == 2


def test_get_from_waiting_queue_with_attention_dp_filtering(
        attention_dp_queue, all_ranks_num_active_requests):
    req1 = RequestQueueItem(
        1,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    req2 = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1,
                                                    attention_dp_relax=True))
    req3 = RequestQueueItem(3,
                            create_mock_request_with_py_schedule_params(
                                attention_dp_rank=None))  # No scheduling params

    attention_dp_queue.waiting_queue.extend([req1, req2, req3])

    # Set rank 0 to full capacity to test filtering
    all_ranks_num_active_requests[0] = 8

    result = attention_dp_queue._get_from_waiting_queue(
        attention_dp_queue.waiting_queue, 3, True,
        all_ranks_num_active_requests)

    assert len(result) == 2
    assert req2 in result
    assert req3 in result
    assert req1 not in result


def test_can_process_attention_dp_request(attention_dp_queue):
    req_no_params = RequestQueueItem(1, Mock())
    assert attention_dp_queue._can_process_attention_dp_request(
        req_no_params, [0, 0, 0, 0]) == True

    req_relax = RequestQueueItem(
        2,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=True))
    assert attention_dp_queue._can_process_attention_dp_request(
        req_relax, [0, 0, 0, 0]) == True

    req_target = RequestQueueItem(
        3,
        create_mock_request_with_py_schedule_params(attention_dp_rank=1,
                                                    attention_dp_relax=False))
    all_ranks = [0, 0, 0, 0]
    assert attention_dp_queue._can_process_attention_dp_request(
        req_target, all_ranks) == True
    assert all_ranks[1] == 1

    req_no_capacity = RequestQueueItem(
        4,
        create_mock_request_with_py_schedule_params(attention_dp_rank=0,
                                                    attention_dp_relax=False))
    all_ranks_full = [8, 0, 0, 0]  # Rank 0 is at capacity
    assert attention_dp_queue._can_process_attention_dp_request(
        req_no_capacity, all_ranks_full) == False


def test_achieve_max_num_active_requests(attention_dp_queue):
    req_list = []
    req_id = 0
    for rank in range(4):
        for i in range(5):
            req_list.append(
                RequestQueueItem(
                    req_id,
                    create_mock_request_with_py_schedule_params(
                        attention_dp_rank=rank, attention_dp_relax=False)))
            req_id += 1
            req_list.append(
                RequestQueueItem(
                    req_id,
                    create_mock_request_with_py_schedule_params(
                        attention_dp_rank=rank, attention_dp_relax=True)))
            req_id += 1

    all_ranks_num_active_requests = [5, 6, 3, 7]
    attention_dp_queue.waiting_queue.extend(req_list)
    available_active_requests = attention_dp_queue.max_num_active_requests * 4 - sum(
        all_ranks_num_active_requests)

    result = attention_dp_queue._get_from_waiting_queue(
        attention_dp_queue.waiting_queue, available_active_requests, True,
        all_ranks_num_active_requests)

    assert len(result) == available_active_requests


def append_to_waiting_queue(waiting_queue, rank, attention_dp_relax):
    req_id = len(waiting_queue)
    waiting_queue.append(
        RequestQueueItem(
            req_id,
            create_mock_request_with_py_schedule_params(
                attention_dp_rank=rank, attention_dp_relax=attention_dp_relax)))


@pytest.mark.parametrize(
    "max_num_active_requests,all_ranks_num_active_requests,request_configs,all_ranks_expected_req_ids",
    [
        # Case: Balanced distribution of relaxed requests
        (3, [0, 0, 0, 0], [(None, True)] * 7, {
            0: [0, 4],
            1: [1, 5],
            2: [2, 6],
            3: [3]
        }),
        # Case: Balanced distribution of relaxed requests
        (3, [1, 2, 3, 0], [(None, True)] * 13, {
            0: [1, 4],
            1: [2],
            2: [],
            3: [0, 3, 5]
        }),
        # Case: Limited by max active
        (3, [0, 0, 0, 0], [(None, True)] * 13, {
            0: [0, 4, 8],
            1: [1, 5, 9],
            2: [2, 6, 10],
            3: [3, 7, 11]
        }),
        # Case: Empty new requests
        (3, [3, 3, 3, 0], [], {
            0: [],
            1: [],
            2: [],
            3: []
        }),
        # Case: Rank 0 is full and cannot schedule attention_dp rank request
        (3, [3, 1, 3, 0], [(0, False), (0, True)], {
            0: [],
            1: [],
            2: [],
            3: [1]
        }),
        # Case: Only room for 1 request, need to skip req0 with attention dp rank
        (3, [3, 2, 3, 3], [(0, False), (0, True)], {
            0: [],
            1: [1],
            2: [],
            3: []
        }),
        # Case: Targeting ranks 1 and 3 that have room
        (3, [2, 1, 3, 0], [(1, False), (3, False)], {
            0: [],
            1: [0],
            2: [],
            3: [1]
        }),
        # Case: Target dp rank specified, by relax is True
        (3, [3, 3, 3, 1], [(0, True), (1, True), (2, True)], {
            0: [],
            1: [],
            2: [],
            3: [0, 1]
        }),
        # Case:
        (3, [3, 3, 3, 0], [(0, False), (1, True), (3, False)], {
            0: [],
            1: [],
            2: [],
            3: [2, 1]
        }),
    ])
def test_attention_dp_scheduling_cases(attention_dp_queue,
                                       max_num_active_requests,
                                       all_ranks_num_active_requests,
                                       request_configs,
                                       all_ranks_expected_req_ids):
    """Test attention DP scheduling with various scenarios."""
    attention_dp_queue.max_num_active_requests = max_num_active_requests

    waiting_queue = deque()
    for rank, relax in request_configs:
        append_to_waiting_queue(waiting_queue, rank, relax)

    run_test_attention_dp_scheduling(attention_dp_queue, waiting_queue,
                                     all_ranks_num_active_requests,
                                     all_ranks_expected_req_ids)


def run_test_attention_dp_scheduling(attention_dp_queue, waiting_queue,
                                     all_ranks_num_active_requests,
                                     all_ranks_expected_req_ids):

    num_ranks = len(all_ranks_num_active_requests)
    total_num_active_requests = sum(all_ranks_num_active_requests)
    total_max_num_active_requests = attention_dp_queue.max_num_active_requests * num_ranks
    enable_attention_dp = True

    new_requests = attention_dp_queue._get_from_waiting_queue(
        waiting_queue,
        total_max_num_active_requests - total_num_active_requests,
        enable_attention_dp, all_ranks_num_active_requests)

    # Schedule attention dp requests
    all_ranks_new_requests = attention_dp_queue._schedule_attention_dp_requests(
        new_requests, all_ranks_num_active_requests)

    assert len(all_ranks_new_requests) == num_ranks
    print("all_ranks_new_requests:", all_ranks_new_requests)
    for rank, reqs in all_ranks_new_requests.items():
        req_ids = [req.id for req in reqs]
        assert req_ids == all_ranks_expected_req_ids[rank]

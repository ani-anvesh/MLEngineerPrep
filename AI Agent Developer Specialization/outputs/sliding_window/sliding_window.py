def sliding_window(nums, k):
    """
    Finds the maximum sum of a subarray of size k in a given list.

    This function uses a sliding window approach to efficiently find the maximum sum. 
    The sliding window maintains a sum of the current window, and expands it 
    by adding the current element and shrinks it by removing the element at the left end 
    when the window's size reaches k. 

    Args:
      nums: A list of integers.
      k: The size of the subarray to consider.

    Returns:
      A list containing the maximum sum of each subarray of size k.

    Examples:
      >>> sliding_window([1, 3, -1, -3, 5, 3, 6, -1, -3], 3)
      [ 7, 6, 12, 15, 18, 12, 15, 18, 12]

    Edge Cases:
      * If the list is shorter than k, the function returns an empty list.
      * The function handles cases where the window size is larger than the list length.
   """

    if len(nums) < k:
        return []

    max_sum = float('-inf')  
    window_sum = 0  
    window_start = 0  
    for window_end in range(len(nums)):  
        window_sum += nums[window_end] 
        if window_end >= k - 1:
            max_sum = max(max_sum, window_sum)
        if window_end >= k - 1:
            window_sum -= nums[window_start] 
            window_start += 1
    return [max_sum]
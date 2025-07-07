def factorial(n):
  """
  Calculates the factorial of a non-negative integer.

  **Description:**
    The factorial of a non-negative integer n, denoted by n!, is the product of all positive integers less than or equal to n. For example, 5! = 5 * 4 * 3 * 2 * 1 = 120.

  **Parameters:**
    n: The non-negative integer for which to calculate the factorial.

  **Return Value:**
    Returns the factorial of n, or 1 if n is 0. 

  **Examples:**
    * factorial(5) == 120
    * factorial(0) == 1
    
  **Edge Cases:**
    * **Non-positive integers:** The function will raise a ValueError if the input is a negative integer.
    * **Zero:** The factorial of 0 is defined as 1.
    * **Large values:** For very large values of n, calculating the factorial can be computationally expensive. 
  """
  if n < 0:
    raise ValueError("Factorial is only defined for non-negative integers")
  elif n == 0:
    return 1
  else:
    return n * factorial(n - 1) 

# Example usage
number = 5
result = factorial(number)
print(f"The factorial of {number} is {result}")
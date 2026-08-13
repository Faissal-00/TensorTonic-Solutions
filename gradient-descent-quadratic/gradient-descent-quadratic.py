def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Set the starting coordinate
    x = x0 
    
    # Loop exactly 'steps' times
    for step in range(steps):
        # Calculate the slope (derivative of ax^2 + bx + c)
        slope = 2 * a * x + b
        
        # Step in the opposite direction of the slope
        x = x - (lr * slope)
        
    # Return the final lowest point found
    return x
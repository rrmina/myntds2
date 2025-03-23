import re
import time
import functools
import sys

import functools
import sys
import traceback

def prefix_print(prefix=""):
    """Decorator that prefixes print statements and errors with a customizable string."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout  # Save original stdout
            original_stderr = sys.stderr  # Save original stderr
            
            class PrintInterceptor:
                def write(self, message):
                    if message.strip():  # Avoid empty lines
                        original_stdout.write(f"{prefix} {message}\n")
                
                def flush(self):  # Needed for interactive environments
                    original_stdout.flush()

            class ErrorInterceptor:
                def write(self, message):
                    if message.strip():
                        original_stderr.write(f"{prefix} ERROR: {message}\n")
                
                def flush(self):
                    original_stderr.flush()

            sys.stdout = PrintInterceptor()  # Redirect stdout
            sys.stderr = ErrorInterceptor()  # Redirect stderr
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = "".join(traceback.format_exception_only(type(e), e))  # Get error message only
                sys.stderr.write(error_msg)
                raise  # Re-raise exception so it behaves normally
            finally:
                sys.stdout = original_stdout  # Restore stdout
                sys.stderr = original_stderr  # Restore stderr
        
        return wrapper
    return decorator

def get_current_time_millis():
    """
    Returns the time in milliseconds since the epoch as an integer number.
    """
    return int(time.time() * 1000)

def sanitize_mlflow_metric_name(name: str) -> str:
    """
    Convert a string to conform to MLflow's metric name rules:
    - Only allows alphanumerics, underscores (_), dashes (-), periods (.), spaces ( ), and slashes (/)
    - Replaces any invalid character with an underscore (_)
    """
    return re.sub(r"[^a-zA-Z0-9_\-./ ]", "_", name)
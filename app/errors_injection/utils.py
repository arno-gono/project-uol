from typing import Any, Callable
import functools
from app.errors_injection.injection_logs import append_failed_injection_logs


def skip_failed_injection(func_error: Callable) -> Callable:
    # Safeguard around every injection function: an error can fail on data an earlier injection corrupted.
    # The error is skipped and recorded rather than stopping the round, so no rule on clashing errors is needed.
    @functools.wraps(func_error)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func_error(*args, **kwargs)

        except Exception as error:
            error_message = f"{type(error).__name__}: {error}"
            print(f"\t\033[91mError {func_error.__name__} could not be injected. {error_message}\033[0m")

            # Every injection function takes the dataframe first and the table name second when it needs one
            if len(args) > 1:
                table_name = args[1]
            elif "table_name" in kwargs:
                table_name = kwargs["table_name"]
            else:
                table_name = ""

            append_failed_injection_logs(
                table_name=table_name,
                error_type=func_error.__name__,
                error_message=error_message
            )
            return None

    return wrapper

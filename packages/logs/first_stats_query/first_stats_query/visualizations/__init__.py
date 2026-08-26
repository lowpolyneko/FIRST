_recipes: dict[str, list] = {}


def recipe(category: str):
    """Register a function as a CLI recipe."""

    def decorator(func):
        _recipes.setdefault(category, []).append((func.__name__, func))
        return func

    return decorator


def get_recipes():
    return _recipes

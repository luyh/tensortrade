from .trading_environment import TradingEnvironment
from .neat_environment import NeatEnvironment

_registry = {

    'basic': {
        'exchange': 'simulated',
        'action_strategy': 'discrete',
        'reward_strategy': 'simple'
    },
    'neat': {
        'exchange': 'simulated',
        'action_strategy':  'discrete',
        'reward_strategy': 'neat'
    }
}


def get(identifier: str) -> TradingEnvironment:
    """Gets the `TradingEnvironment` that matches with the identifier.

    Arguments:
        identifier: The identifier for the `TradingEnvironment`

    Raises:
        KeyError: if identifier is not associated with any `TradingEnvironment`
    """
    if identifier not in _registry.keys():
        raise KeyError(
            'Identifier {} is not associated with any `TradingEnvironment`.'.format(identifier))
    return TradingEnvironment(**_registry[identifier])

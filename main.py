import logging

import hydra

import experiment
from experiment import ExpBase

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="main", version_base="1.1")
def main(config):
    exp: ExpBase = getattr(experiment, config.exp.name)(config)
    exp.run()


if __name__ == "__main__":
    main()

    
    print('落ちんポッポ')       

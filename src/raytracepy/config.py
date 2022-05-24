from dataclasses import dataclass


@dataclass
class Config:
    disable_all_warnings = False
    single_warning = False

    def __post_init__(self):
        self._warning_ogrid = False if self.disable_all_warnings else True

    @property
    def warning_ogrid(self):
        cur_value = self._warning_ogrid
        if self.single_warning:
            self._warning_ogrid = False
        return cur_value


config = Config()

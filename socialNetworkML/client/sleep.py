#!/usr/bin/python3
# sleep infinitely long

import time
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

if __name__ == "__main__":
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    exit(0)

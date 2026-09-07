"""DR-Sight ML backend package.

Smart India Hackathon 2026 prototype (PS SIH26038, Team Diasight).
Screening aid only - not a certified medical device.
"""

import os

# albumentations otherwise makes a blocking network call on import to check for
# a newer version, which spams an SSL warning on offline / locked-down hosts.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

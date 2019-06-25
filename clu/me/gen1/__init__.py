from .v1 import V1
from .n1 import N1
from .n5 import N5
from .n6 import N6
from .n7 import N7

from clu.me.vae.vae1 import VAE1
from clu.me.vae.vae2 import VAE2
from clu.me.vae.vae3 import VAE3
from clu.me.vae.vae4 import VAE4
from clu.me.vae.vae5 import VAE5

name2m_class = {v.__name__: v for v in [N6, VAE1, VAE2, VAE3, VAE4, VAE5]}

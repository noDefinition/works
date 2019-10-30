# from clu.me.gen1.n1 import N1
from clu.me.gen1.n5 import N5
# from clu.me.gen1.n6 import N6
from clu.me.vae.vae1 import VAE1
from clu.me.vae.vae2 import VAE2
# from clu.me.gen2.sbx import SBX
from clu.me.ae.ae_spectral import AeSpectral
from clu.me.ae.ae_spectral_2 import AeSpectral2
from clu.me.ae.ae_spectral_3 import AeSpectral3
from clu.me.ae.ae_base import AEBase
from clu.me.ae.ae_lstm import AELstm
from clu.me.ae.ae_soft import AeSoft

name2m_class = {
    v.__name__: v for v in [N5, VAE1, VAE2, AeSpectral, AeSpectral2, AeSpectral3, AELstm, AeSoft]
}

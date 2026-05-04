##online.py
import numpy as np
from Functions.Algo2 import algorithm_2

def initialize_model(mean_waveform, noise_variance, config):
    
    if mean_waveform is None:
        raise ValueError("No waveforms detected → cannot initialize model")

    n_sources = config["n_sources"]
    #t_R = config["t_R"]

    H0 = mean_waveform
    P0 = np.diag((0.1 * np.abs(np.concatenate(H0)))**2) ## revisar si ya se concateno aa ?

    #t_0 modificar apra 2 sources
    #t0 = np.array([np.random.uniform(2000, 2500)]) # t_0 = np.array([np.random.uniform(2000, 2500)]) np.random.randint(t_R+1, 3*t_R+1, size=n_sources)
    
    # t0 = np.array([
    # np.random.uniform(2700, 2700),   # pour le neurone "positif" habla de nurone ??
    # np.random.uniform(12000, 12000)  # pour le neurone "négatif"
    # ])
    # beta = np.array([
    #     np.random.uniform(3,3),     
    #     np.random.uniform(2,2)       
    # ])
    # brta tambien adaptar para 2 sources ya que beta y t-0 entre sources no es la misma, apra los que serian alfa y beta
    #beta = np.random.uniform(1, 2, size=n_sources) # beta = np.array([np.random.uniform(1,2)])

    #v = 8 # Esto tenia paul apra inicializar el ruido 
    return {
        "H": H0,
        "P": P0,
        "v": noise_variance
    }


def run_online(signal, offline_results, config):

    #breakpoint()
    model = initialize_model(offline_results["mean_waveforms_per_mu"], offline_results["noise_variance"], config)

    # 🔥 Y = señal observada
    Y = signal

    # parámetros
    n_MU = config["n_sources"]
    t_R = config["t_R"]
    
    if isinstance(model["H"], (list, tuple)):
        # Modelo con múltiples MUAPs
        
        params_per_mu = len(model["H"][0])  # Longitud de cada MUAP (40)
        #n_mus = len(model["H"])              # Número de MUs (2)
        ell_RI = params_per_mu     # Total: 80
    else:
        # Modelo con un solo MUAP (array simple)
        ell_RI = len(model["H"])  
    # Ahora (lista/array)
    ell_infinity = [100 * i for i in t_R]
    n_s = config["n_s"]  # puedes moverlo a config luego
    # plt.plot(model["H"][0])
    # plt.show()
    # plt.plot(model["H"][1])
    # plt.show()
    
    U_est, Y_est, H_est, Theta_est, theta_history = algorithm_2(
        Y,
        n_MU,
        t_R,
        ell_RI,
        n_s,
        ell_infinity,
        H0=model["H"],
        P0=model["P"],
        t_0= offline_results["t0_init"], #model["t"],
        beta= offline_results["beta_init"], #model["beta"],
        v=model["v"],
        config = config
    )
    
    return {
        "Y": Y,
        "U_est": U_est,
        "Y_est": Y_est,
        "H_est": H_est,
        "Theta_est": Theta_est,
        "theta_history": theta_history
    }
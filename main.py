import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk

from scipy.signal import medfilt, welch
from sklearn.preprocessing import StandardScaler
from matplotlib.widgets import Button as pltButton

folder_path = "dati"
fs = 128

#Array direzioni e stazioni
#East
eastACAE = []
eastACAEnorm = np.array([])
eastPIS1 = []
eastPIS1norm = np.array([])
eastRITE = []
eastRITEnorm = np.array([])
eastSOLO = []
eastSOLOnorm = np.array([])
eastSTRZ = []
eastSTRZnorm = np.array([])
#North
northACAE = []
northACAEnorm = np.array([])
northPIS1 = []
northPIS1norm = np.array([])
northRITE = []
northRITEnorm = np.array([])
northSOLO = []
northSOLOnorm = np.array([])
northSTRZ = []
northSTRZnorm = np.array([])
#Up
upACAE = []
upACAEnorm = np.array([])
upPIS1 = []
upPIS1norm = np.array([])
upRITE = []
upRITEnorm = np.array([])
upSOLO = []
upSOLOnorm = np.array([])
upSTRZ = []
upSTRZnorm = np.array([])

# Funzione per caricare i dati da file .rtl (se sono CSV-like)
def load_rtl_files(folder_path):
    signals = {}

    #Colonne da salvare per East, North ed Up
    cols_to_save = [3,5,10]
    for filename in os.listdir(folder_path):
        if filename.endswith(".rtl"):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, usecols=cols_to_save, header=None)
            signals[filename] = df
    return signals

# Funzione per calcolare lo spettro del segnale (equivalente a pspectrum)
def compute_spectrum(signal, fs=128):  # fs è la frequenza di campionamento ipotetica
    f, Pxx = welch(signal, fs=fs, nperseg=1024)
    return f, Pxx

def fillMatrices(signals):
    #Riempimento matrici
    for key in signals.keys():
        if "ACAE" in key:
            for i in signals.get(key).iloc[:,0].to_numpy():
                eastACAE.append(i)
            for i in signals.get(key).iloc[:,1].to_numpy():
                northACAE.append(i)
            for i in signals.get(key).iloc[:,2].to_numpy():
                upACAE.append(i)
        if "PIS1" in key:
            for i in signals.get(key).iloc[:,0].to_numpy():
                eastPIS1.append(i)
            for i in signals.get(key).iloc[:,1].to_numpy():
                northPIS1.append(i)
            for i in signals.get(key).iloc[:,2].to_numpy():
                upPIS1.append(i)
        if "RITE" in key:
            for i in signals.get(key).iloc[:,0].to_numpy():
                eastRITE.append(i)
            for i in signals.get(key).iloc[:,1].to_numpy():
                northRITE.append(i)
            for i in signals.get(key).iloc[:,2].to_numpy():
                upRITE.append(i)
        if "SOLO" in key:
            for i in signals.get(key).iloc[:,0].to_numpy():
                eastSOLO.append(i)
            for i in signals.get(key).iloc[:,1].to_numpy():
                northSOLO.append(i)
            for i in signals.get(key).iloc[:,2].to_numpy():
                upSOLO.append(i)
        if "STRZ" in key:
            for i in signals.get(key).iloc[:,0].to_numpy():
                eastSTRZ.append(i)
            for i in signals.get(key).iloc[:,1].to_numpy():
                northSTRZ.append(i)
            for i in signals.get(key).iloc[:,2].to_numpy():
                upSTRZ.append(i)

# Funzione per elaborare e plottare il segnale
def plot_signals(original, denoised, fs=128, window_name = ""):
    # Creazione della figura con 3 subplot
    plt.figure(window_name,figsize=(10, 5))

    # Subplot 1: Segnale originale vs denoised
    plt.subplot(3, 1, 1)
    plt.plot(original, label="Noised", alpha=0.7)
    plt.plot(denoised, label="Denoised", linestyle="dashed")
    plt.legend()
    plt.title("Segnale Originale e Denoised")

    # Subplot 2: Spettro del segnale
    f1, Pxx1 = compute_spectrum(original, fs)
    f2, Pxx2 = compute_spectrum(denoised, fs)
    plt.subplot(3, 1, 2)
    plt.semilogy(f1, Pxx1, label="Noised")
    plt.semilogy(f2, Pxx2, label="Denoised", linestyle="dashed")
    plt.legend()
    plt.title("Spettro del Segnale")

    # Subplot 3: Filtro Mediano
    plt.subplot(3, 1, 3)
    #plt.plot(original, label="Noised", alpha=0.7)
    #plt.plot(denoised, label="Denoised")
    plt.plot(medfilt(original, kernel_size=5), label="Noised Filtered")
    plt.plot(medfilt(denoised, kernel_size=5), label="Denoised Filtered")
    plt.legend()
    plt.title("Filtro Mediano")

    plt.tight_layout()
    plt.show()

#######------------------------Noise_Reduction--------------------------------------#######
def nr_tau(seg, ncomp, p, tau, fc):
    """
    Riduzione del Rumore con PCA
    
    Parametri:
    seg   -- matrici del segnale 
    ncomp -- numero di componenti in input
    p     -- numero di componenti principali (default: 2)
    tau   -- parametro tau
    fc    -- frequenza di campionamento
    
    Output:
    y_tot -- segnale elaborato
    """

    nseg = np.array(seg)

    y_tot = []
    out = []

    for i in range(1): 
        yseg = nseg[:]
        xseg = np.arange(len(yseg)) / fc
        
        lr = 0.0001
        alpha = 1 
        
        # Normalizzazione (simile a mapstd di Matlab)
        scaler = StandardScaler()
        yseg = scaler.fit_transform(yseg.reshape(-1, 1)).flatten()
        
        # Rete neurale
        w, npatt, yseg = inNR_tau(ncomp, p, yseg, tau)
        
        maxit = 50
        freq_sup = fc / 2
        
        # Funzione training
        w, f, stima, epoca = eq9(yseg, lr, w, p, npatt, ncomp, 1, xseg, alpha, freq_sup, maxit)

        # Funzione visualizzazione
        xseg, ynew, outd = visualization(w, xseg, yseg, npatt, ncomp, i, len(seg), tau)

        # Normalizzazione
        y_tot.append(scaler.fit_transform(ynew.reshape(-1, 1)).flatten())
        out.append(outd)
    
    y_tot = np.array(y_tot)
    y_tot = y_tot[0,:]
    return y_tot

def inNR_tau(ncomp, p, aa, tau):
    """
    Inizializza i pesi della rete neurale per la riduzione del rumore.

    Parametri:
    ncomp -- numero di componenti input
    p     -- numero di componenti principali
    aa    -- segnale di input
    tau   -- parametro per l'inizializzazione del pattern

    Output:
    w     -- matrice con pesi inizializzati
    npatt -- numero di pattern
    aa    -- segnale di input modificato
    """
    npatt = aa.shape[0]
    npatt = (npatt - ncomp) // 1

    w = np.zeros((ncomp, p))

    for s in range(p):
        c = 0
        for r in range(0, ncomp, tau):
            w[c, s] = aa[r + s]
            c += 1
    
    return w, npatt, aa

def eq9(aa, lr, w, p, npatt, ncomp, funz, x, alpha, intsup, maxit):
    """
    Generalizzazione regola di Oja

    Parametri:
    aa     -- segnale di input
    lr     -- tasso di apprendimento
    w      -- matrice dei pesi iniziale
    p      -- numero di componenti principali
    npatt  -- numero di pattern
    ncomp  -- numero di componenti in input
    funz   -- tipo di funzione (1 per la tangente iperbolica, altrimenti logaritmica)
    x      -- dati di input
    alpha  -- fattore di scaling
    intsup -- supporto in frequenza
    maxit  -- numero massimo di iterazioni

    Output:
    w      -- matrice dei pesi aggiornata
    f      -- risposta in frequenza
    stima  -- stima ottenuta
    epoca  -- numero di epoche
    """
    testnorm = 1.0
    
    I = np.eye(ncomp)
    epoca = 0
    flagtot = 0
    w = w / np.linalg.norm(w)
    wold = w.copy()
    
    while testnorm > 0.01 and epoca < maxit:
        k = 0
        
        while k <= npatt:
            xup = aa[k:ncomp + k].reshape(ncomp, 1)
            
            if funz == '1':
                y = np.tanh(alpha * xup.T @ w)
            else:
                y = np.sign(xup.T @ w) * np.log(1 + np.abs(alpha * xup.T @ w))
            
            w = w + lr * ((I - w @ w.T) @ xup @ y)
            w = w / np.linalg.norm(w)
            
            k += 1

        # Necessario per far funzionare le operazioni su w (SVD per inversa stabile)
        U, S, Vt = np.linalg.svd(w @ w.T)
        S_inv = np.diag(1 / (S + 1e-6))
        w_inv = U @ S_inv @ Vt
        w = np.real(w_inv @ w)
        
        flagtot, f, stima = visfrStima(w, ncomp, npatt, p, x, intsup)
        
        testnorm = sum(np.sqrt(np.sum((wold[:, i] - w[:, i]) ** 2)) for i in range(p))

        print(f'Epoca: {epoca} Testnorm: {testnorm}')
        wold = w.copy()
        
        epoca += 1
    
    return w, f, stima, epoca

def visfrStima(w, ncomp, npatt, p, x, intsup):
    """
    Stima della risposta in frequenza utilizzando l'algoritmo MUSIC.

    Parametri:
    w      -- matrice dei pesi
    ncomp  -- numero di componenti
    npatt  -- numero di pattern
    p      -- numero di componenti principali
    x      -- dati di input
    intsup -- limite superiore di frequenza

    Output:
    flagtot -- flag che indica il successo dell'operazione (1 se avvenuta con successo)
    f       -- valori di frequenza
    stima   -- densità spettrale di potenza stimata
    """
    flagtot = 0
    f = np.arange(0.001, intsup, 0.1)

    stima = 10 * np.log10(musicStima(w, f, x, npatt))
    
    plt.figure(2)
    plt.plot(f, stima, 'b')
    plt.title('Algoritmo Stima(seq_2): errore')
    plt.xlabel('Frequency')
    plt.ylabel('Psd')
    plt.pause(0.1)
    
    if np.any(stima > 0.0): 
        flagtot = 1
    
    return flagtot, f, stima

def musicStima(w, f, x, npatt):
    """
    Stimatore di frequenza MUSIC.

    Parametri:
    w     -- matrice dei pesi
    f     -- frequenze da analizzare
    x     -- dati di input
    npatt -- numero di pattern

    Output:
    y     -- spettro di frequenza stimato
    """
    m, p = w.shape

    sommac = 0.0
    sommas = 0.0
    
    for i in range(p):
        somcos = 0.0
        somsen = 0.0

        somcos = np.sum(w[:, i] * np.cos(2 * np.pi * f[:, None] * x[:m]), axis=1)
        somsen = np.sum(w[:, i] * np.sin(2 * np.pi * f[:, None] * x[:m]), axis=1)
        
        sommac += somcos ** 2
        sommas += somsen ** 2
    
    somma = sommac + sommas
    y = 1.0 / (m - somma)
    
    return y

def visualization(w, xseg, yseg, npatt, ncomp, ind, tot, tau):
    k = 0
    ynew = np.zeros(npatt + ncomp)
    outd = np.zeros((npatt + 1, 2))

    while k <= npatt:
        yseg_k = yseg[k:k + ncomp]

        if yseg_k.size == ncomp:
            yseg_k = yseg_k.reshape(ncomp, 1)  
        else:
            print(f"Errore: yseg_k ha {yseg_k.size} elementi invece di {ncomp}")
            yseg_k = np.zeros((ncomp, 1))
        
        ynew[k:k + ncomp] = (w @ (w.T @ yseg_k)).flatten()

        outd[k, :] = (w.T @ yseg_k).flatten()
        
        k += 1
    
    return xseg, ynew, outd
#######-----------------------------------------------------------------------------#######

#GRAPHICS
def back(event):
    plt.close("all")

#ACAE
def on_button0_click():
    plt.plot(eastACAE, label='EAST ACAE')
    plt.legend()

    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(eastACAEbutton_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()

def on_button1_click():
    plt.plot(northACAE, label='NORTH ACAE')
    plt.legend()

    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(northACAEbutton_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()

def on_button2_click():
    plt.plot(upACAE, label='UP ACAE')
    plt.legend()

    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(upACAEbutton_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()
    
def eastACAEbutton_clicked(event):
    print("ncomp = ", ncomp, "p = ", p, "tau = ", tau)

    eastACAEdenoised = nr_tau(eastACAE,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(eastACAE, eastACAEdenoised, fs, "east ACAE")

def northACAEbutton_clicked(event):
    northACAEdenoised = nr_tau(northACAE,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(northACAE, northACAEdenoised, fs, "north ACAE")

def upACAEbutton_clicked(event):
    upACAEdenoised = nr_tau(upACAE,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(upACAE, upACAEdenoised, fs, "up ACAE")

#PIS1
def on_button3_click():
    plt.plot(eastPIS1, label='EAST PIS1')
    plt.legend()
    
    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(eastPIS1button_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()

def on_button4_click():
    plt.plot(northPIS1, label='NORTH PIS1')
    plt.legend()
    
    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(northPIS1button_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()

def on_button5_click():
    plt.plot(upPIS1, label='UP PIS1')
    plt.legend()
    
    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(upPIS1button_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)
    
    plt.show()
    
def eastPIS1button_clicked(event):
    eastPIS1denoised = nr_tau(eastPIS1,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(eastPIS1, eastPIS1denoised, fs, "EAST PIS1")

def northPIS1button_clicked(event):
    northPIS1denoised = nr_tau(northPIS1,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(northPIS1, northPIS1denoised, fs, "NORTH PIS1")

def upPIS1button_clicked(event):
    upPIS1denoised = nr_tau(upPIS1,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(upPIS1, upPIS1denoised, fs, "UP PIS1")

#RITE
def on_button6_click():
    plt.plot(eastRITE, label='EAST RITE')
    plt.legend()

    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(eastRITEbutton_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()

def on_button7_click():
    plt.plot(northRITE, label='NORTH RITE')
    plt.legend()
    
    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(northRITEbutton_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()

def on_button8_click():
    plt.plot(upRITE, label='UP RITE')
    plt.legend()
    
    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(upRITEbutton_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)
    
    plt.show()
    
def eastRITEbutton_clicked(event):
    eastRITEdenoised = nr_tau(eastRITE,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(eastRITE, eastRITEdenoised, fs, "EAST RITE")

def northRITEbutton_clicked(event):
    northRITEdenoised = nr_tau(northRITE,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(northRITE, northRITEdenoised, fs, "NORTH RITE")

def upRITEbutton_clicked(event):
    upRITEdenoised = nr_tau(upRITE,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(upRITE, upRITEdenoised, fs, "UP RITE")

#SOLO
def on_button9_click():
    plt.plot(eastSOLO, label='EAST SOLO')
    plt.legend()

    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(eastSOLObutton_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()
   
def on_button10_click():
    plt.plot(northSOLO, label='NORTH SOLO')
    plt.legend()

    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(northSOLObutton_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()
   
def on_button11_click():
    plt.plot(upSOLO, label='UP SOLO')
    plt.legend()

    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(upSOLObutton_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()

def eastSOLObutton_clicked(event):
    eastSOLOdenoised = nr_tau(eastSOLO,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(eastSOLO, eastSOLOdenoised, fs, "EAST SOLO")

def northSOLObutton_clicked(event):
    northSOLOdenoised = nr_tau(northSOLO,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(northSOLO, northSOLOdenoised, fs, "NORTH SOLO")

def upSOLObutton_clicked(event):
    upSOLOdenoised = nr_tau(upSOLO,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(upSOLO, upSOLOdenoised, fs, "UP SOLO")

#STRZ
def on_button12_click():
    plt.plot(eastSTRZ, label='EAST STRZ')
    plt.legend()

    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(eastSTRZbutton_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()

def on_button13_click():
    plt.plot(northSTRZ, label='NORTH STRZ')
    plt.legend()

    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(northSTRZbutton_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()

def on_button14_click():
    plt.plot(upSTRZ, label='UP STRZ')
    plt.legend()

    cont_button = pltButton(plt.axes([0.8, 0.01, 0.15, 0.05]), 'Continue')        
    cont_button.on_clicked(upSTRZbutton_clicked)

    exit_button = pltButton(plt.axes([0.65, 0.01, 0.15, 0.05]), 'Exit')
    exit_button.on_clicked(back)

    plt.show()

def eastSTRZbutton_clicked(event):
    eastSTRZdenoised = nr_tau(eastSTRZ,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(eastSTRZ, eastSTRZdenoised, fs, "EAST STRZ")

def northSTRZbutton_clicked(event):
    northSTRZdenoised = nr_tau(northSTRZ,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(northSTRZ, northSTRZdenoised, fs, "NORTH STRZ")

def upSTRZbutton_clicked(event):
    upSTRZdenoised = nr_tau(upSTRZ,ncomp,p,tau,fs) 
    plt.close("all")
    plot_signals(upSTRZ, upSTRZdenoised, fs, "UP STRZ")

#######-----------------------------------------------------------------------------#######

#def create_window():
    
#######-----------------------------------------------------------------------------#######
signals = load_rtl_files(folder_path)
fillMatrices(signals)

ncomp = 128
p = 2
tau = 1

#Normalizzazione
scaler = StandardScaler()

#East
eastACAE = np.array(eastACAE)
eastACAE = scaler.fit_transform(eastACAE.reshape(-1, 1)).flatten()
eastPIS1 = np.array(eastPIS1)
eastPIS1 = scaler.fit_transform(eastPIS1.reshape(-1, 1)).flatten()
eastRITE = np.array(eastRITE)
eastRITE = scaler.fit_transform(eastRITE.reshape(-1, 1)).flatten()
eastSOLO = np.array(eastSOLO)
eastSOLO = scaler.fit_transform(eastSOLO.reshape(-1, 1)).flatten()
eastSTRZ = np.array(eastSTRZ)
eastSTRZ = scaler.fit_transform(eastSTRZ.reshape(-1, 1)).flatten()

#NORTH
northACAE = np.array(northACAE)
northACAE = scaler.fit_transform(northACAE.reshape(-1, 1)).flatten()
northPIS1 = np.array(northPIS1)
northPIS1 = scaler.fit_transform(northPIS1.reshape(-1, 1)).flatten()
northRITE = np.array(northRITE)
northRITE = scaler.fit_transform(northRITE.reshape(-1, 1)).flatten()
northSOLO = np.array(northSOLO)
northSOLO = scaler.fit_transform(northSOLO.reshape(-1, 1)).flatten()
northSTRZ = np.array(northSTRZ)
northSTRZ = scaler.fit_transform(northSTRZ.reshape(-1, 1)).flatten()

#UP
upACAE = np.array(upACAE)
upACAE = scaler.fit_transform(upACAE.reshape(-1, 1)).flatten()
upPIS1 = np.array(upPIS1)
upPIS1 = scaler.fit_transform(upPIS1.reshape(-1, 1)).flatten()
upRITE = np.array(upRITE)
upRITE = scaler.fit_transform(upRITE.reshape(-1, 1)).flatten()
upSOLO = np.array(upSOLO)
upSOLO = scaler.fit_transform(upSOLO.reshape(-1, 1)).flatten()
upSTRZ = np.array(upSTRZ)
upSTRZ = scaler.fit_transform(upSTRZ.reshape(-1, 1)).flatten()

# Creazione della finestra principale
root = tk.Tk()
root.title("Main")

def on_ncomp_entry_modified(*args):
    global ncomp
    ncomp = int(ncomp_entry.get())
    
def on_p_entry_modified(*args):
    global p
    p = int(p_entry.get())

def on_tau_entry_modified(*args):
    global tau
    tau = int(tau_entry.get())

#VARS
ncomp_var = tk.StringVar()
ncomp_var.trace_add("write", on_ncomp_entry_modified)
p_var = tk.StringVar()
p_var.trace_add("write", on_p_entry_modified)
tau_var = tk.StringVar()
tau_var.trace_add("write", on_tau_entry_modified)

# Frame
frameACAE = tk.Frame()
frameACAE.pack(pady=25)
framePIS1 = tk.Frame()
framePIS1.pack(pady=25)
frameRITE = tk.Frame()
frameRITE.pack(pady=25)
frameSOLO = tk.Frame()
frameSOLO.pack(pady=25)
frameSTRZ = tk.Frame()
frameSTRZ.pack(pady=25)

frameIntInput = tk.Frame()
frameIntInput.pack(pady=25)

# ACAE STATION
ACAElabel = tk.Label(frameACAE, text="ACAE station", font=("Arial", 12))
ACAElabel.pack(pady=15)

button0 = tk.Button(frameACAE, 
                    text="EAST ACAE", 
                    command=on_button0_click)
button1 = tk.Button(frameACAE, 
                    text="NORTH ACAE", 
                    command=on_button1_click)
button2 = tk.Button(frameACAE, 
                    text="UP ACAE", 
                    command=on_button2_click)
    
button0.pack(side=tk.LEFT,padx=10)
button1.pack(side=tk.LEFT,padx=10)
button2.pack(side=tk.LEFT,padx=10)

#PIS1 Station
PIS1label = tk.Label(framePIS1, text="PIS1 station", font=("Arial", 12))
PIS1label.pack(pady=15)

button3 = tk.Button(framePIS1, 
                    text="EAST PIS1", 
                    command=on_button3_click)
button4 = tk.Button(framePIS1, 
                    text="NORTH PIS1", 
                    command=on_button4_click)
button5 = tk.Button(framePIS1, 
                    text="UP PIS1", 
                    command=on_button5_click)

button3.pack(side=tk.LEFT,padx=10)
button4.pack(side=tk.LEFT,padx=10)
button5.pack(side=tk.LEFT,padx=10)

#RITE Station
RITElabel = tk.Label(frameRITE, text="RITE station", font=("Arial", 12))
RITElabel.pack(pady=15)

button6 = tk.Button(frameRITE, 
                    text="EAST RITE", 
                    command=on_button6_click)
button7 = tk.Button(frameRITE, 
                    text="NORTH RITE", 
                    command=on_button7_click)
button8 = tk.Button(frameRITE, 
                    text="UP RITE", 
                    command=on_button8_click)

button6.pack(side=tk.LEFT,padx=10)
button7.pack(side=tk.LEFT,padx=10)
button8.pack(side=tk.LEFT,padx=10)

#SOLO
SOLOlabel = tk.Label(frameSOLO, text="SOLO station", font=("Arial", 12))
SOLOlabel.pack(pady=15)
button9 = tk.Button(frameSOLO, 
                    text="EAST SOLO", 
                    command=on_button9_click)
button10 = tk.Button(frameSOLO, 
                    text="NORTH SOLO", 
                    command=on_button10_click)
button11 = tk.Button(frameSOLO, 
                    text="UP SOLO", 
                    command=on_button11_click)

button9.pack(side=tk.LEFT,padx=10)
button10.pack(side=tk.LEFT,padx=10)
button11.pack(side=tk.LEFT,padx=10)

#STRZ
STRZlabel = tk.Label(frameSTRZ, text="STRZ station", font=("Arial", 12))
STRZlabel.pack(pady=15)
button12 = tk.Button(frameSTRZ, 
                    text="EAST STRZ", 
                    command=on_button12_click)
button13 = tk.Button(frameSTRZ, 
                    text="NORTH STRZ", 
                    command=on_button13_click)
button14 = tk.Button(frameSTRZ,
                    text="UP STRZ", 
                    command=on_button14_click)

button12.pack(side=tk.LEFT,padx=10)
button13.pack(side=tk.LEFT,padx=10)
button14.pack(side=tk.LEFT,padx=10)

#INTINPUT
InputLabel = tk.Label(frameIntInput, text="Input Denoising", font=("Arial", 12))
InputLabel.grid(row=0, column=0, columnspan=3)

ncomp_label = tk.Label(frameIntInput, text="Components:", font=("Arial", 12))
ncomp_label.grid(row=1, column=0)
ncomp_entry = tk.Entry(frameIntInput, width=10, textvariable=ncomp_var)
ncomp_entry.insert(0, "128")
ncomp_entry.grid(row=2, column=0)

p_label = tk.Label(frameIntInput, text="P:", font=("Arial", 12))
p_label.grid(row=1, column=1)
p_entry = tk.Entry(frameIntInput, width=10)
p_entry.insert(0, "2")
p_entry.grid(row=2, column=1)

tau_label = tk.Label(frameIntInput, text="Tau:", font=("Arial", 12))
tau_label.grid(row=1, column=2)
tau_entry = tk.Entry(frameIntInput, width=10)
tau_entry.insert(0, "1")
tau_entry.grid(row=2, column=2)

# Avvio dell'interfaccia
root.mainloop()

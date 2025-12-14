from datetime import datetime

import essentia.standard as es
import matplotlib.pyplot as plt
import numpy as np


name = str(input("Indique cómo quiere guardar la canción de la que va a extraer su huella digital: "))

# Ruta del archivo de audio
audio_path = "/mnt/c/Users/Oriol/Music/4K YouTube to MP3/Claude Debussy - La Mer.mp3"

# Cargar el audio y lo transforma en mono
loader = es.MonoLoader(filename=audio_path)
audio = loader()


# -------- PASO 1: Forma de onda --------
plt.figure(figsize=(12, 4))
plt.plot(audio)
plt.title(f"Forma de onda del audio {name}")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
# Nombre de archivo con fecha y hora
file_name = datetime.now().strftime(
    f'/home/oriol/audio-fingerprinting/forma_de_onda_del_audio_{name}_%Y%m%d_%H%M%S.png'
)
plt.savefig(file_name)
print(f"Imagen guardada en: {file_name}")


# -------- PASO 2: Espectrograma --------

# Parámetros de STFT
frame_size = 2048
hop_size = 512

window = es.Windowing(type='hann')
spectrum = es.Spectrum()

# Calculamos espectrograma
spectrogram = []

for frame in es.FrameGenerator(audio, 
                               frameSize=frame_size, 
                               hopSize=hop_size, 
                               startFromZero=True
    ):  #Divide el audio en frames

    mag = spectrum(window(frame))  #Aplica una función de suavizamiento (window)
    spectrogram.append(mag)

spectrogram = np.array(spectrogram).T   # Hay que transponerla para que la gráfica quede "frecuencia-tiempo"

# Graficar espectrograma
plt.figure(figsize=(12, 6))
plt.imshow(10 * np.log10(spectrogram + 1e-6), aspect='auto', origin='lower')  #Este cálculo convierte la potencia de cada frecuencia a decibelios
plt.title(f"Espectrograma de {name} (STFT)")
plt.xlabel("Frames")
plt.ylabel("Frecuencia (bins)")
plt.colorbar(label="dB")
file_name_1 = datetime.now().strftime(
    f'/home/oriol/audio-fingerprinting/espectrograma_{name}_STFT_%Y%m%d_%H%M%S.png'
)
plt.savefig(file_name_1)
print(f"Imagen guardada en: {file_name_1}")

# -------- PASO 3: Detección de picos (peaks) --------
from scipy.ndimage import maximum_filter

# Parámetros de detección
neighborhood = 20          # Tamaño del filtro para buscar máximos locales
threshold_ratio = 0.5      # Relación respecto al máximo global

# Filtro de máximos locales
local_max = maximum_filter(spectrogram, size=neighborhood) == spectrogram  #Máscara booleana de máximos locales
# Umbral para evitar ruido
amplitude_threshold = threshold_ratio * spectrogram.max()

# Coordenadas de picos significativos
peak_coords = np.where(local_max & (spectrogram > amplitude_threshold)) #Discrimina los máximos no significativos (50% de intenstidad máxima)

# Convertimos a lista de (freq_bin, time_frame)
peaks = list(zip(peak_coords[0], peak_coords[1])) #zip combina las dos filas del array y forma tuplas de puntos del tipo "frecuencia-tiempo"

print(f"Total de picos detectados: {len(peaks)}")

# Visualización de los picos sobre el espectrograma
plt.figure(figsize=(12, 6))
plt.imshow(10 * np.log10(spectrogram + 1e-6), aspect='auto', origin='lower')
plt.scatter(peak_coords[1], peak_coords[0], s=2, c='red')
plt.title(f"Picos detectados en el espectrograma de {name}")
plt.xlabel("Frames")
plt.ylabel("Frecuencia (bins)")
file_name_2 = datetime.now().strftime(
    f'/home/oriol/audio-fingerprinting/picos_detectados_{name}_STFT_%Y%m%d_%H%M%S.png'
)
plt.savefig(file_name_2)
print(f"Imagen guardada en: {file_name_2}")

# -------- PASO 4: Crear hashes a partir de los picos --------

def generate_hashes(peaks, fanout=5, max_time_delta=50):
    """
    Genera hashes de audio a partir de picos espectrales.

    Parámetros
    ----------
    peaks : list of tuple (int, int)
        Lista de picos (frecuencia, tiempo).
    fanout : int
        Número de picos posteriores a combinar.
    max_time_delta : int
        Máxima separación temporal permitida entre picos.

    Salida
    -------
    list of tuple (int, int)
        Lista de hashes y su tiempo de referencia.
    """
    hashes = []

    # Ordenar picos por tiempo (2º elemento de la tupla)
    peaks_sorted = sorted(peaks, key=lambda x: x[1]) 

    for i in range(len(peaks_sorted)):
        f1, t1 = peaks_sorted[i]

        # Emparejar con los siguientes picos cercanos
        for j in range(1, fanout):
            if i + j < len(peaks_sorted):
                f2, t2 = peaks_sorted[i + j]
                delta_t = t2 - t1

                # Ignorar si están muy lejos en el tiempo
                if 0 < delta_t <= max_time_delta:
                    h = hash((f1, f2, delta_t))
                    hashes.append((h, t1))

    return hashes


# Generar fingerprints
fingerprints = generate_hashes(peaks)

print(f"Total de hashes generados para {name}: {len(fingerprints)}")
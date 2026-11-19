import os
import time
import sounddevice as sd
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import scipy.fftpack as fft
from scipy.io import wavfile
import matplotlib.pyplot as plt

# --- Parametri ---
SAMPLE_RATE = 44100     # Rata de esantionare standard
BLOCK_SIZE = 1024       # Dimensiunea bufferului pentru timp real
DURATION = 10           # Durata testului demonstrativ (secunde)

# Intervalul pe care il pastram (banda vocii)
FREQ_MIN = 150.0        # Taie bubuiturile/zumzetul de joasa frecventa
FREQ_MAX = 3500.0       # Taie fasaitul ascutit de inalta frecventa

# Filtru Butterworth (elimina complet zornaitul dintre blocuri)
sos = butter(N=4, Wn=[FREQ_MIN, FREQ_MAX], btype='bandpass', fs=SAMPLE_RATE, output='sos')
zi = sosfilt_zi(sos)

raw_recorded_blocks = []
cleaned_recorded_blocks = []

def audio_callback(indata, outdata, frames, time_info, status):
    global zi
    if status:
        print(f"Status buffer: {status}")

    signal = indata[:, 0].copy()
    raw_recorded_blocks.append(signal)

    # Filtrare continua in timp real
    cleaned_signal, zi = sosfilt(sos, signal, zi=zi)
    cleaned_recorded_blocks.append(cleaned_signal)

    # Redare in casti in timp real
    outdata[:, 0] = cleaned_signal.astype(np.float32)

def main():
    # Determinam folderul exact in care se afla acest script .py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_brut = os.path.join(script_dir, "audio_brut.wav")
    path_filtrat = os.path.join(script_dir, "audio_filtrat.wav")

    print("=" * 60)
    print(f" TEST TIMP REAL PORNIT ({DURATION} secunde)")
    print(" Pune castile! Vorbeste la microfon sau bate in masa/fa un fasait.")
    print("=" * 60)

    try:
        with sd.Stream(samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=1,
                       dtype='float32', callback=audio_callback):
            for s in range(DURATION, 0, -1):
                print(f"Inregistrare & filtrare in timp real... mai sunt {s} secunde")
                time.sleep(1)
    except Exception as e:
        print(f"\n[EROARE AUDIO]: Nu s-a putut porni microfonul/placa de sunet: {e}")
        print("Verifica in Windows 11: Settings -> Privacy & security -> Microphone.")
        return

    if len(raw_recorded_blocks) == 0:
        print("\n[EROARE]: Microfonul nu a trimis date! Verifica setarile audio din Windows.")
        return

    print("\nProcesare incheiata! Salvare fisiere si generare grafice...")

    # Imbinam blocurile captate
    raw_audio = np.concatenate(raw_recorded_blocks)
    cleaned_audio = np.concatenate(cleaned_recorded_blocks)

    # Scalare la int16 si salvare
    raw_int16 = (raw_audio * 32767).clip(-32768, 32767).astype(np.int16)
    cleaned_int16 = (cleaned_audio * 32767).clip(-32768, 32767).astype(np.int16)

    wavfile.write(path_brut, SAMPLE_RATE, raw_int16)
    wavfile.write(path_filtrat, SAMPLE_RATE, cleaned_int16)

    print("-> Fisiere salvate cu succes la aceste adrese exacte:")
    print(f"   1. {path_brut}")
    print(f"   2. {path_filtrat}")

    # --- Generare Grafice Laborator ---
    time_axis = np.linspace(0, len(raw_audio) / SAMPLE_RATE, len(raw_audio))

    raw_fft_full = np.abs(fft.fft(raw_audio))[:len(raw_audio) // 2]
    cleaned_fft_full = np.abs(fft.fft(cleaned_audio))[:len(cleaned_audio) // 2]
    freq_axis_full = np.fft.fftfreq(len(raw_audio), d=1.0 / SAMPLE_RATE)[:len(raw_audio) // 2]

    plt.figure(figsize=(11, 7))

    plt.subplot(2, 2, 1)
    plt.plot(time_axis[:2000], raw_audio[:2000], color='red')
    plt.title("Semnal Brut cu Zgomot (Domeniul Timp)")
    plt.xlabel("Timp (s)")
    plt.ylabel("Amplitudine")
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(time_axis[:2000], cleaned_audio[:2000], color='green')
    plt.title("Semnal Filtrat (Domeniul Timp)")
    plt.xlabel("Timp (s)")
    plt.ylabel("Amplitudine")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(freq_axis_full, raw_fft_full, color='pink')
    plt.title("Spectru Frecvente Brut (Inainte de filtrare)")
    plt.xlabel("Frecventa (Hz)")
    plt.ylabel("Magnitudine")
    plt.xlim(0, 8000)
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(freq_axis_full, cleaned_fft_full, color='blue')
    plt.title(f"Spectru dupa Filtrare [{FREQ_MIN} - {FREQ_MAX} Hz]")
    plt.xlabel("Frecventa (Hz)")
    plt.ylabel("Magnitudine")
    plt.xlim(0, 8000)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
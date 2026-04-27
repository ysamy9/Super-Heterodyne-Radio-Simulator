import numpy as np
import tkinter as tk
from tkinter import ttk
import scipy.io.wavfile as wav
from scipy import signal
import sounddevice as sd
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

#Parameters
FS       = 44100
FACTOR   = 12
FS_SIM   = FS * FACTOR
F_IF     = 25000
DURATION = 10
CARRIERS = [100000, 150000, 200000]

#Gain and distance
AGC_MAX_GAIN = 80.0
PRESETS = [10, 50, 150]


SNR_LEVELS = [
    ("No noise",     999,  999),   
    ("noise Lv1",  10,   50),   
    ("noise Lv2",   -5,   50),  
]


#------------------------

def load_wav(file, start=0):
    try:
        fs, data = wav.read(file)
        data = data[int(start * fs): int((start + DURATION) * fs)]
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32) / (np.max(np.abs(data)) + 1e-9)
        return signal.resample(data, len(data) * FACTOR)
    except Exception:
        return np.zeros(int(FS_SIM * DURATION), dtype=np.float32)


def distance_attenuation(d_km):
    return 1.0 / max(d_km, 0.1)


def station_power(fdm_1sec, fc):
    bpf = signal.sosfilt(
        signal.butter(4, [fc - 8000, fc + 8000], 'bandpass', fs=FS_SIM, output='sos'),
        fdm_1sec
    )
    return float(np.mean(bpf ** 2))

def safe_agc_gain(rx_power, target_power):
   
    g = np.sqrt(target_power / (rx_power + 1e-12))
    return float(min(g, AGC_MAX_GAIN))


#Modulation

def mod_am(m, fc, t):
    return (1 + 0.8 * m) * np.cos(2 * np.pi * fc * t)

def mod_dsb_lc(m, fc, t):
    return (1 + 0.9 * m) * np.cos(2 * np.pi * fc * t)

def mod_nbfm(m, fc, t):
    kf = 5000
    return np.cos(2 * np.pi * fc * t + 2 * np.pi * kf * np.cumsum(m) / FS_SIM)

#Noise

def _scale_noise(noise_arr, signal_arr, snr_db):
    sig_power   = float(np.mean(signal_arr ** 2))
    noise_power = float(np.mean(noise_arr ** 2))
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-12))
    return noise_arr * scale

def make_inband_noise(white_noise, fc):
    sos = signal.butter(8, [fc - 8000, fc + 8000], 'bandpass', fs=FS_SIM, output='sos')
    return signal.sosfiltfilt(sos, white_noise.astype(np.float64)).astype(np.float32)


#demodulation 

def _rf_and_if(fdm, t, fc, rf_on, lo_offset):
    rf = signal.sosfilt(
        signal.butter(6, [fc - 8000, fc + 8000], 'bandpass', fs=FS_SIM, output='sos'),
        fdm
    ) if rf_on else fdm
    lo    = np.cos(2 * np.pi * (fc + F_IF + lo_offset) * t)
    mixed = rf * lo
    return signal.sosfilt(
        signal.butter(6, [F_IF - 6000, F_IF + 6000], 'bandpass', fs=FS_SIM, output='sos'),
        mixed
    )

_DC_BLOCK = signal.butter(2, 30, 'highpass', fs=FS, output='sos')

def demodulate_am(fdm, t, fc, rf_on=True, lo_offset=0):
    if_sig   = _rf_and_if(fdm, t, fc, rf_on, lo_offset)
    baseband = if_sig * np.cos(2 * np.pi * F_IF * t)
    lpf      = signal.sosfilt(signal.butter(6, 7000, 'lowpass', fs=FS_SIM, output='sos'), baseband)
    return signal.sosfilt(_DC_BLOCK, lpf[::FACTOR])

def demodulate_fm(fdm, t, fc, rf_on=True, lo_offset=0):
    if_sig = _rf_and_if(fdm, t, fc, rf_on, lo_offset)
    diff   = np.diff(if_sig, prepend=if_sig[0])
    env    = np.abs(signal.hilbert(diff))
    audio  = env - np.mean(env)
    lpf    = signal.sosfilt(signal.butter(6, 7000, 'lowpass', fs=FS_SIM, output='sos'), audio)
    return signal.sosfilt(_DC_BLOCK, lpf[::FACTOR])

def demodulate(fdm, t, fc, rf_on=True, lo_offset=0):
    if fc == CARRIERS[2]:
        return demodulate_fm(fdm, t, fc, rf_on, lo_offset)
    return demodulate_am(fdm, t, fc, rf_on, lo_offset)

#Plots

def _plot_spectrum(ax, s, title, color, xlim):
    f   = np.fft.rfftfreq(len(s), 1 / FS_SIM)
    mag = np.abs(np.fft.rfft(s))
    ax.plot(f / 1000, mag, color=color, lw=0.7)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(xlim)
    mask = (f / 1000 >= xlim[0]) & (f / 1000 <= xlim[1])
    if mask.any():
        ax.set_ylim(0, max(np.percentile(mag[mask], 99) * 1.3, 1e-6))
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)

#  Application 

class RadioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Super-Heterodyne Radio")
        self.root.geometry("900x760")

        self.freq       = tk.DoubleVar(value=CARRIERS[0])
        self.rf_on      = tk.BooleanVar(value=True)
        self.lo_off     = tk.DoubleVar(value=0.0)
        self.snr_level  = tk.IntVar(value=0)
        self.filter_off = tk.BooleanVar(value=False)

        self.dist = [float(PRESETS[0])] * 3

        self._gain_cache  = [1.0, 1.0, 1.0]
        self._target_power = 1.0   

        self.is_on  = False
        self.stream = None
        self.ptr    = 0

        self._prepare()
        self._build_gui()
        self._update_display()   

#prepare signal

    def _prepare(self):
        msgs = [load_wav('station1.wav', 9),
                load_wav('station2.wav', 1),
                load_wav('station3.wav', 0)]
        length = max(len(m) for m in msgs)
        self.msgs = [np.pad(m, (0, length - len(m))) for m in msgs]
        self.t    = np.arange(length) / FS_SIM

        ref_fdm = (
            mod_am    (self.msgs[0], CARRIERS[0], self.t) +
            mod_dsb_lc(self.msgs[1], CARRIERS[1], self.t) +
            mod_nbfm  (self.msgs[2], CARRIERS[2], self.t)
        ) / 3.0
        self._target_power = float(np.mean(
            [station_power(ref_fdm[:FS_SIM], fc) for fc in CARRIERS]
        ))

       
        rng   = np.random.default_rng(seed=0)
        white = rng.standard_normal(length).astype(np.float32)
        self._white_noise  = white
        self._inband_noise = {fc: make_inband_noise(white, fc) for fc in CARRIERS}

        self._rebuild_fdm()

    def _rebuild_fdm(self):
        
        self.fdm_clean = (
            mod_am    (self.msgs[0], CARRIERS[0], self.t) * distance_attenuation(self.dist[0]) +
            mod_dsb_lc(self.msgs[1], CARRIERS[1], self.t) * distance_attenuation(self.dist[1]) +
            mod_nbfm  (self.msgs[2], CARRIERS[2], self.t) * distance_attenuation(self.dist[2])
        ) / 3.0
        self._refresh_gain_cache()

    def _refresh_gain_cache(self):
        snap = self.fdm_clean[:FS_SIM]
        for i, fc in enumerate(CARRIERS):
            rx_pw = station_power(snap, fc)
            self._gain_cache[i] = safe_agc_gain(rx_pw, self._target_power)


    def _station_index(self):
        fc = self.freq.get()
        return int(np.argmin([abs(fc - c) for c in CARRIERS]))

    def _get_fdm(self):
      
        idx = self.snr_level.get()
        _, wb_snr, ib_snr = SNR_LEVELS[idx]
        if wb_snr >= 999:
            return self.fdm_clean   # no noise at all

        fc  = self.freq.get()
        fc_key = CARRIERS[int(np.argmin([abs(fc - c) for c in CARRIERS]))]

        if not self.filter_off.get():
            noise = _scale_noise(self._inband_noise[fc_key], self.fdm_clean, ib_snr)
        else:
            noise = _scale_noise(self._white_noise, self.fdm_clean, wb_snr)

        return self.fdm_clean + noise


    def _set_distance(self, station_idx, d_km):
        self.dist[station_idx] = float(d_km)
        self._rebuild_fdm()
        for j, btn in enumerate(self._dist_btns[station_idx]):
            active = (PRESETS[j] == d_km)
            btn.config(relief="sunken" if active else "flat",
                       bg="#003300"   if active else "#1a1a1a")


    def _update_display(self):
        idx     = self._station_index()
        fc      = CARRIERS[idx]
        snap    = self.fdm_clean[:FS_SIM]
        rx_pw   = station_power(snap, fc)
        gain    = self._gain_cache[idx]
        rx_db   = 10 * np.log10(rx_pw   + 1e-12)
        gain_db = 20 * np.log10(gain    + 1e-12)
        self.lbl_rx_power.config(text=f"Rx Power:  {rx_db:+.1f} dB")
        self.lbl_agc_gain.config(text=f"AGC Gain:  {gain_db:+.1f} dB")
        self.root.after(400, self._update_display)

#Audio

    def _toggle_power(self):
        if not self.is_on:
            self.is_on = True
            self.btn_power.config(text="Power: ON", bg="#00ff00", fg="black")
            self.stream = sd.OutputStream(
                samplerate=FS, channels=1,
                callback=self._audio_cb, blocksize=2048)
            self.stream.start()
        else:
            self.is_on = False
            self.btn_power.config(text="Power: OFF", bg="#ff0000", fg="white")
            if self.stream:
                self.stream.stop()
                self.stream.close()

    def _audio_cb(self, out, frames, time_info, status):
        n   = frames * FACTOR
        end = self.ptr + n
        if end > len(self.fdm_clean):
            self.ptr = 0
            end = n

        fc      = self.freq.get()
        idx     = self._station_index()
        chunk   = self._get_fdm()[self.ptr:end]
        t_chunk = self.t[self.ptr:end]

        chunk = chunk * self._gain_cache[idx]

        audio = demodulate(chunk, t_chunk, fc, self.rf_on.get(), self.lo_off.get())

        out[:frames] = np.clip(audio[:frames], -1.0, 1.0).reshape(-1, 1)
        self.ptr = end


    def _plot_baseband(self):
        fig, axs = plt.subplots(3, 2, figsize=(12, 9))
        plt.subplots_adjust(hspace=0.6, wspace=0.35)
        colors = ['#00bfff', '#00e676', '#ff9800']
        labels = ['Station 1 (AM — DSB-LC,  ka=0.8)',
                  'Station 2 (DSB-LC,  ka=0.9)',
                  'Station 3 (NBFM,  kf=5000 Hz)']
        for i in range(3):
            sl = self.msgs[i][FS_SIM // 10: FS_SIM // 10 + FS_SIM // 20]
            axs[i, 0].plot(sl, color=colors[i], lw=0.7)
            axs[i, 0].set_title(f"{labels[i]} — Time Domain", fontsize=9)
            axs[i, 0].set_xlabel("Sample")
            axs[i, 0].set_ylabel("Amplitude")
            axs[i, 0].grid(True, alpha=0.3)
            _plot_spectrum(axs[i, 1], self.msgs[i][:20000],
                           f"{labels[i]} — Spectrum", colors[i], (0, 15))
        plt.suptitle("Baseband Message Signals", fontsize=13, fontweight='bold')
        plt.show()

    def _plot_stages(self):
        fc  = self.freq.get()
        n   = int(0.1 * FS_SIM)
        fdm = self._get_fdm()[:n]
        t   = self.t[:n]
        rf  = signal.sosfilt(signal.butter(6, [fc - 8000, fc + 8000],
                             'bandpass', fs=FS_SIM, output='sos'), fdm)
        mix = rf * np.cos(2 * np.pi * (fc + F_IF) * t)
        iff = signal.sosfilt(signal.butter(6, [F_IF - 6000, F_IF + 6000],
                             'bandpass', fs=FS_SIM, output='sos'), mix)
        if fc == CARRIERS[2]:
            diff = np.diff(iff, prepend=iff[0])
            bb   = np.abs(signal.hilbert(diff)); bb -= np.mean(bb)
            bb_label = "FM Discriminator output"
        else:
            bb       = iff * np.cos(2 * np.pi * F_IF * t)
            bb_label = "Baseband (raw)"
        lpf  = signal.sosfilt(signal.butter(6, 7000, 'lowpass', fs=FS_SIM, output='sos'), bb)
        fc_k = fc / 1000
        fig, axs = plt.subplots(3, 2, figsize=(13, 9))
        plt.subplots_adjust(hspace=0.65, wspace=0.35)
        for ax, s, title, color, xlim in [
            (axs[0, 0], fdm, "FDM Airwave",                      '#00e676', (80, 220)),
            (axs[1, 0], rf,  f"After RF BPF ({fc_k:.0f} kHz)",  '#ff5252', (fc_k - 20, fc_k + 20)),
            (axs[2, 0], mix, "After Mixer",                       '#ff9800', (0, 300)),
            (axs[0, 1], iff, "After IF BPF (25 kHz)",            '#ff5252', (15, 35)),
            (axs[1, 1], bb,  bb_label,                            '#00bfff', (0, 60)),
            (axs[2, 1], lpf, "Baseband after LPF",               '#b39ddb', (0, 12)),
        ]:
            _plot_spectrum(ax, s, title, color, xlim)
        plt.suptitle(f"Receiver Stages — Tuned to {fc_k:.0f} kHz", fontsize=13, fontweight='bold')
        plt.show()

    def _plot_rf_bypass(self):
        fc  = self.freq.get()
        n   = int(0.1 * FS_SIM)
        fdm = self._get_fdm()[:n]
        t   = self.t[:n]
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for ax, sig, title, color in [
            (axs[0], demodulate(fdm, t, fc, rf_on=True),  "With RF BPF (clean)",           '#00bfff'),
            (axs[1], demodulate(fdm, t, fc, rf_on=False), "Without RF BPF (interference)", '#ff5252'),
        ]:
            _plot_spectrum(ax, sig, title, color, (0, 12))
            ax.set_title(title, fontsize=10, fontweight='bold')
        plt.suptitle(f"RF Stage Bypass — Tuned to {fc / 1000:.0f} kHz",
                     fontsize=13, fontweight='bold')
        plt.show()

    def _plot_lo_offset(self):
        fc  = self.freq.get()
        n   = int(0.1 * FS_SIM)
        fdm = self._get_fdm()[:n]
        t   = self.t[:n]
        fig, axs = plt.subplots(1, 3, figsize=(14, 5))
        plt.subplots_adjust(wspace=0.4)
        for ax, offset, color, label in zip(
            axs,
            [0, 100, 1000],
            ['#00e676', '#ff9800', '#ff5252'],
            ['No offset', '+0.1 kHz offset', '+1.0 kHz offset']
        ):
            _plot_spectrum(ax, demodulate(fdm, t, fc, lo_offset=float(offset)),
                           label, color, (0, 12))
            ax.set_title(label, fontsize=10, fontweight='bold')
        plt.suptitle(f"LO Frequency Offset — Tuned to {fc / 1000:.0f} kHz",
                     fontsize=13, fontweight='bold')
        plt.show()

    def _plot_noise(self):
        fc    = self.freq.get()
        n     = int(0.1 * FS_SIM)
        idx   = self.snr_level.get()
        _, wb_snr, ib_snr = SNR_LEVELS[idx]
        fc_key = CARRIERS[self._station_index()]

        clean = self.fdm_clean[:n]

        if wb_snr >= 999:
            noisy_off = clean
            noisy_on  = clean
            snr_label = "no noise"
        else:
            wb_noise = _scale_noise(self._white_noise[:n], self.fdm_clean[:n], wb_snr)
            ib_noise = _scale_noise(self._inband_noise[fc_key][:n], self.fdm_clean[:n], ib_snr)
            noisy_off = clean + wb_noise
            noisy_on  = clean + ib_noise
            snr_label = f"wb={wb_snr} dB / ib={ib_snr} dB"

        fig, axs = plt.subplots(1, 3, figsize=(14, 5))
        plt.subplots_adjust(wspace=0.4)
        for ax, s, title, color, xlim in [
            (axs[0], clean,     "",                          '#00e676', (80, 220)),
            (axs[1], noisy_off, "",       '#ff5252', (80, 220)),
            (axs[2], noisy_on,  "",    '#00bfff',
             (fc / 1000 - 30, fc / 1000 + 30)),
        ]:
            _plot_spectrum(ax, s, title, color, xlim)
        plt.suptitle(f"Pre-selector Effect — {fc/1000:.0f} kHz  |  {snr_label}",
                     fontsize=13, fontweight='bold')
        plt.show()


    def _refresh_fspec(self):
        
        idx = self.snr_level.get()
        _, wb_snr, ib_snr = SNR_LEVELS[idx]
        filt_active = not self.filter_off.get()
        if wb_snr >= 999:
            spec = "No noise "
        elif filt_active:
            spec = ("  order 8    "
                    "rolloff 96   output SNR ≈ 47 dB  (inaudible)")
        else:
            spec = (f"  channel SNR = {wb_snr} dB  →  "
                    f"output SNR ≈ {'~20 dB  ' if wb_snr >= 5 else '  '}")
        self.lbl_fspec.config(text=spec)

    def _build_gui(self):
        try:
            img = Image.open("bg.png").resize((1000, 800))
            self._bg = ImageTk.PhotoImage(img)
            tk.Label(self.root, image=self._bg).place(x=0, y=0, relwidth=1, relheight=1)
        except Exception:
            self.root.configure(bg='black')

        style = ttk.Style()
        style.theme_use('default')
        style.configure("G.Horizontal.TScale", troughcolor="black", background="#00ff00",
                        bordercolor="#00ff00", lightcolor="#00ff00", darkcolor="#00ff00")
        style.configure("R.Horizontal.TScale", troughcolor="black", background="#ff5252",
                        bordercolor="#ff5252", lightcolor="#ff5252", darkcolor="#ff5252")

        tk.Label(self.root, text="Super-Heterodyne Radio", bg="black", fg="#00ff00",
                 font=("Courier New", 18, "bold"), pady=8).pack(fill="x")

        c = tk.Frame(self.root, bg="black")
        c.place(relx=0.5, rely=0.5, anchor="center")

        self.btn_power = tk.Button(
            c, text="Power: OFF", font=("Arial", 12, "bold"),
            bg="#ff0000", fg="white", width=22, relief="flat",
            command=self._toggle_power)
        self.btn_power.pack(pady=6)

#Station selector

        pf = tk.Frame(c, bg="black"); pf.pack(pady=4)
        for txt, freq in [("Station 1\n100 kHz AM",     CARRIERS[0]),
                          ("Station 2\n150 kHz DSB-LC", CARRIERS[1]),
                          ("Station 3\n200 kHz NBFM",   CARRIERS[2])]:
            tk.Button(pf, text=txt, bg="#00ff00", fg="black",
                      font=("Courier New", 8, "bold"), width=13, height=2, relief="flat",
                      command=lambda f=freq: self.freq.set(f)).pack(side="left", padx=5)

        ttk.Scale(c, from_=80000, to=220000, variable=self.freq,
                  length=520, style="G.Horizontal.TScale").pack(pady=4)
        self.lbl_freq = tk.Label(c, text="100.0 kHz", fg="#00ff00", bg="black",
                                 font=("Consolas", 22, "bold"), width=14)
        self.lbl_freq.pack(pady=4)
        self.freq.trace_add("write", lambda *_: self.lbl_freq.config(
            text=f"{self.freq.get() / 1000:.1f} kHz"))

        tk.Frame(c, bg="#00ff00", height=1, width=520).pack(pady=5)




        tk.Label(c, text="STATION DISTANCE", bg="black", fg="#00ff00",
                 font=("Courier New", 10, "bold")).pack(pady=(4, 2))

        ST_COLORS = ["#00bfff", "#00e676", "#ff9800"]
        ST_NAMES  = ["Station 1", "Station 2", "Station 3"]
        self._dist_btns = [[], [], []]

        grid = tk.Frame(c, bg="black"); grid.pack(pady=2)

        tk.Label(grid, text="", bg="black", width=10).grid(row=0, column=0)
        for j, lbl in enumerate(["Near ", "Mid ", "Far "]):
            tk.Label(grid, text=lbl, bg="black", fg="#666666",
                     font=("Courier New", 8), width=12).grid(row=0, column=j + 1, padx=4)

        for i in range(3):
            tk.Label(grid, text=ST_NAMES[i], bg="black", fg=ST_COLORS[i],
                     font=("Courier New", 9), anchor="w", width=10
                     ).grid(row=i + 1, column=0, pady=3)
            for j, d_km in enumerate(PRESETS):
                btn = tk.Button(
                    grid, text=f"{d_km} km",
                    bg="#003300" if j == 0 else "#1a1a1a",   
                    fg="#00ff00", font=("Courier New", 9, "bold"),
                    width=9, relief="sunken" if j == 0 else "flat",
                    command=lambda si=i, dk=d_km: self._set_distance(si, dk))
                btn.grid(row=i + 1, column=j + 1, padx=4, pady=2)
                self._dist_btns[i].append(btn)



        agc_row = tk.Frame(c, bg="black"); agc_row.pack(pady=6)
        self.lbl_rx_power = tk.Label(
            agc_row, text="Rx Power:  -- dB", bg="black", fg="#aaaaaa",
            font=("Courier New", 11), width=20)
        self.lbl_rx_power.pack(side="left", padx=16)
        self.lbl_agc_gain = tk.Label(
            agc_row, text="AGC Gain:  -- dB", bg="black", fg="#00ff00",
            font=("Courier New", 11), width=20)
        self.lbl_agc_gain.pack(side="left", padx=16)

        tk.Frame(c, bg="#00ff00", height=1, width=520).pack(pady=5)



        tk.Label(c, text="CHANNEL NOISE", bg="black", fg="#ff5252",
                 font=("Courier New", 10, "bold")).pack(pady=(4, 2))

        snr_frame = tk.Frame(c, bg="black"); snr_frame.pack(pady=2)
        for i, (lbl, wb, ib) in enumerate(SNR_LEVELS):
            tk.Radiobutton(
                snr_frame, text=lbl, variable=self.snr_level, value=i,
                bg="black", fg="#ff5252", selectcolor="black",
                activebackground="black", activeforeground="#ff5252",
                font=("Courier New", 10)
            ).pack(side="left", padx=14)

        self.lbl_fspec = tk.Label(c, text="", bg="black", fg="#555555",
                                  font=("Courier New", 8))
        self.lbl_fspec.pack(pady=1)
        self.snr_level.trace_add("write",  lambda *_: self._refresh_fspec())
        self.filter_off.trace_add("write", lambda *_: self._refresh_fspec())

        frow = tk.Frame(c, bg="black"); frow.pack(pady=2)
        tk.Checkbutton(
            frow,
            text="Disable filter  —  raw noise reaches demodulator",
            variable=self.filter_off,
            bg="black", fg="#ff5252", selectcolor="black",
            activebackground="black", activeforeground="#ff5252",
            font=("Courier New", 9)
        ).pack(side="left", padx=8)
        tk.Frame(c, bg="#00ff00", height=1, width=520).pack(pady=5)




        bf = tk.Frame(c, bg="black"); bf.pack(pady=4)
        for txt, cmd in [("Baseband\nPlots",  self._plot_baseband),
                         ("Stage\nPlots",     self._plot_stages),
                         ("RF\nBypass",       self._plot_rf_bypass),
                         ("LO\nOffset",       self._plot_lo_offset),
                         ("Noise\nPlot",      self._plot_noise)]:
            tk.Button(bf, text=txt, bg="#1a1a1a", fg="#00ff00",
                      font=("Courier New", 9, "bold"), width=10, height=2, relief="flat",
                      activebackground="#003300", activeforeground="#00ff00",
                      command=cmd).pack(side="left", padx=4)

        tk.Frame(c, bg="#00ff00", height=1, width=520).pack(pady=5)




        qf = tk.Frame(c, bg="black"); qf.pack(pady=2)
        tk.Label(qf, text="RF Stage:", bg="black", fg="#aaaaaa",
                 font=("Courier New", 10)).pack(side="left", padx=(0, 6))
        tk.Checkbutton(qf, text="Enabled", variable=self.rf_on, bg="black", fg="#00ff00",
                       selectcolor="black", activebackground="black", activeforeground="#00ff00",
                       font=("Courier New", 10)).pack(side="left")

        q5f = tk.Frame(c, bg="black"); q5f.pack(pady=2)
        tk.Label(q5f, text="LO Offset:", bg="black", fg="#aaaaaa",
                 font=("Courier New", 10)).pack(side="left", padx=(0, 6))
        for txt, val in [("None", 0.0), ("+0.1 kHz", 100.0), ("+1 kHz", 1000.0)]:
            tk.Radiobutton(q5f, text=txt, variable=self.lo_off, value=val,
                           bg="black", fg="#00ff00", selectcolor="black",
                           activebackground="black", activeforeground="#00ff00",
                           font=("Courier New", 10)).pack(side="left", padx=5)

        tk.Label(self.root,
                 text="         Supervision: Dr Doaa Gamal | Dr Samar Mokhtar  ",
                 bg="black", fg="#444444", font=("Courier New", 9)
                 ).pack(side="bottom", fill="x", pady=4)


if __name__ == "__main__":
    root = tk.Tk()
    app  = RadioApp(root)
    root.mainloop()

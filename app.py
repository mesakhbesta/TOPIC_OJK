import streamlit as st
import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import joblib
from bertopic import BERTopic
from huggingface_hub import hf_hub_download

st.title("Email Complaint Processing and Topic Classification")
st.write("Upload file Excel atau CSV dan lihat distribusi topik beserta full teks Cleaned Complaint.")

# File uploader
uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file is not None:
    # Load file berdasarkan ekstensi
    if uploaded_file.name.endswith("xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.success("File berhasil diupload!")
    
    # ----------------------- Data Cleaning Pipeline -----------------------
    # Step 1: Remove 'Dear Bapak/Ibu Helpdesk OJK' from Notes
    def remove_dear_ojk(text):
        if isinstance(text, str):
            return re.sub(r"Dear\s*Bapak/Ibu\s*Helpdesk\s*OJK", "", text)
        return text

    df['Notes'] = df['Notes'].apply(remove_dear_ojk)

    # Step 2: Extract the complaint from the email body
    def extract_complaint(text):
        if not isinstance(text, str):
            return "Bagian komplain tidak ditemukan."
        
        start_patterns = [
            r"PERHATIAN: E-mail ini berasal dari pihak di luar OJK.*?attachment.*?link.*?yang terdapat pada e-mail ini."
        ]
        
        end_patterns = [
            r"(From\s*.*?From|Best regards|Salam|Atas perhatiannya|Regards|Best Regards|Mohon\s*untuk\s*melengkapi\s*data\s*.*tabel\s*dibawah).*",
            r"From:\s*Direktorat\s*Pelaporan\s*Data.*"  
        ]
        
        start_match = None
        for pattern in start_patterns:
            matches = list(re.finditer(pattern, text, re.DOTALL))
            if matches:
                start_match = matches[-1].end()  # End of matched start pattern
        
        if start_match:
            text = text[start_match:].strip()  # Remove content before start pattern
        
        for pattern in end_patterns:
            end_match = re.search(pattern, text, re.DOTALL)
            if end_match:
                text = text[:end_match.start()].strip()  # Remove content after end pattern
        
        sensitive_info_patterns = [
            r"Nama\s*Terdaftar\s*.*",
            r"Email\s*.*",
            r"No\.\s*Telp\s*.*",
            r"User\s*Id\s*/\s*User\s*Name\s*.*",
            r"No\.\s*KTP\s*.*",
            r"Nama\s*Perusahaan\s*.*",
            r"Nama\s*Pelapor\s*.*",
            r"No\.\s*Telp\s*Pelapor\s*.*",
            r"Internal",
            r"Dengan\s*hormat.*",
            r"Jenis\s*Usaha\s*.*",
            r"Keterangan\s*.*",
            r"No\.\s*SK\s*.*",
            r"Alamat\s*website/URL\s*.*",
            r"Selamat\s*(Pagi|Siang|Sore).*",
            r"Kepada\s*Yth\.\s*Bapak/Ibu.*",
            r"On\s*full_days\s*\d+\d+,\s*\d{4}-\d{2}-\d{2}\s*at\s*\d{2}:\d{2}.*",
            r"Dear\s*Bapak/Ibu\s*Helpdesk\s*OJK.*",
            r"No\.\s*NPWP\s*Perusahaan\s*.*",
            r"Aplikasi\s*OJK\s*yang\s*di\s*akses\s*.*",
            r"Yth\s*.*",
            r"demikian\s*.*",
            r"Demikian\s*.*",
            r"Demikianlah\s*.*"
        ]
        
        for pattern in sensitive_info_patterns:
            text = re.sub(pattern, "", text)
        
        return text if text else "Bagian komplain tidak ditemukan."

    df['Complaint'] = df.apply(lambda row: extract_complaint(row['Notes']), axis=1)

    # Step 3: Combine 'Complaint' and 'Summary' and apply cut-off cleaning
    def cut_off_general(complaint):
        cut_off_keywords = [
            "PT Mandiri Utama FinanceSent: Wednesday, November 6, 2024 9:11 AMTo",
            "Atasdan kerjasama, kami ucapkan h Biro Hukum dan KepatuhanPT Jasa Raharja (Persero)Jl. HR Rasuna Said Kav. C-2 12920Jakarta Selatan",
            "h._________", 
            "h Imawan FPT ABC Multifinance Pesan File Kirim (PAPUPPK/2024-12-31/Rutin/Gagal)Kotak MasukTelusuri semua pesan berlabel Kotak MasukHapus",
            "KamiBapak/Ibu untuk pencerahannya", 
            "kami ucapkan h. ,", 
            "-- , DANA PENSIUN BPD JAWA", 
            "sDian PENYANGKALAN.",
            "------------------------Dari: Adrian", 
            "hormat saya RidwanForwarded", 
            "--h, DANA PENSIUN WIJAYA",
            "Mohon InfonyahKantor", 
            "an arahannya dari Bapak/ Ibu", 
            "ya untuk di check ya.Thank", 
            "Kendala:Thank youAddelin",
            ",Sekretaris DAPENUrusan Humas & ProtokolTazkya", 
            "Mohon arahannya.Berikut screenshot", 
            "Struktur_Data_Pelapor_IJK_(PKAP_EKAP)_-_Final_2024", 
            "Annie Clara DesiantyComplianceIndonesia",
            "Dian Rosmawati RambeCompliance", 
            "Beararti apakah,Tri WahyuniCompliance DeptPT.", 
            "Dengan alamat email yang didaftarkan",
            "dan arahan", 
            ",AJB Bumiputera", 
            "’h sebelumnya Afriyanty", 
            "PENYANGKALAN.", 
            "h Dana Pensiun PKT",
            ", h , Tasya PT.", 
            "Contoh: 10.00", 
            "hAnnisa Adelya SerawaiPT Fazz", 
            "sebagaimana gambar di bawah ini",
            "PT Asuransi Jiwa SeaInsure On Fri", 
            "hJana MaesiatiBanking ReportFinance", 
            "Tembusan", 
            "Sebagai referensi", 
            "hAdriansyah",
            "h atas bantuannya Dwi Anggina", 
            "PT Asuransi Jiwa SeaInsure", 
            "dengan notifikasi dibawah ini", 
            "Terima ksh",
            ": DISCLAIMER", 
            "Sebagai informasi", 
            "nya. h.Kind s,Melati", 
            ": DISCLAIMER", 
            "Petugas AROPT", 
            "h,Julianto", 
            "h,Hernawati",
            "Dana Pensiun Syariah", 
            ",Tria NoviatyStrategic"
        ]
        
        for keyword in cut_off_keywords:
            if keyword in complaint:
                complaint = complaint.split(keyword)[0]
        return complaint

    df['Complaint'] = df['Complaint'].apply(cut_off_general)
    df['Complaint'] = df['Complaint'] + " " + df['Summary']

    # Optional: Remove unwanted header or subject in Complaint
    subject_pattern = r"(?i)Subject:\s*(Re:\s*FW:|RE:|FW:|PTAsuransiAllianzUtamaIndonesiaPT Asuransi Allianz Utama Indonesia)?\s*"
    df['Complaint'] = df['Complaint'].str.replace(subject_pattern, "", regex=True)

    # Step 4: Clean the complaint text (lowercase, remove non-alphabet characters, stopwords)
    stopword_factory = StopWordRemoverFactory()
    stopwords = stopword_factory.get_stop_words()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = ' '.join([word for word in text.split() if word not in stopwords])
        return text

    df['Cleaned_Complaint'] = df['Complaint'].apply(clean_text)

    # Step 5: Remove unwanted words using regex replacements on Cleaned_Complaint
    words_to_remove = r"\b(lapor|client|jasa|web|rencana|bisnis|pengawasan|dokumen|asuransi|rencana|permohonan|indonesia|allianz|keuangan|otoritas|no|penggunaan|antifraud|penerapan|strategi|fraud|anti|realisasi|saf|tersebut|nya|data|terdapat|periode|melalui|perusahaan|sesuai|melakukan|hak|komplek|laporan|pelaporan|modul|apolo|sebut|terap|email|pt|mohon|sampai|ikut|usaha|dapat|tahun|kini|lalu|kendala|ojk|laku|guna|aplikasi|atas|radius|prawiro|jakarta pusat)\b"
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.lower()
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(words_to_remove, "", regex=True)

    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\bmasuk\b", "login", regex=True)
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\blog-in\b", "login", regex=True)
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\brenbis\b", "rencana bisnis", regex=True)
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\bapkap\b", "ap kap", regex=True)
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\baro\b", "administrator responsible officer", regex=True)
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\bro\b", "responsible officer", regex=True)

    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].str.replace(r"\s+", " ", regex=True).str.strip()

    # Step 6: Final Clean-up
    df['Cleaned_Complaint'] = df['Cleaned_Complaint'].apply(lambda x: clean_text(x))

    # Step 7: Display the final result
    st.write("### Final Cleaned Complaint Data")
    st.dataframe(df[['Incident Number', 'Summary', 'Cleaned_Complaint']].head(10))

    # ----------------------- Classification -----------------------
    st.write("### Topic Classification")

    # Load BERTopic models from Kaggle
    repo_id = 'mesakhbesta/NLP_OJK'
    model_utama = hf_hub_download(repo_id=repo_id, filename='bertopic_utama.joblib')
    model_utama = joblib.load(model_utama)
    model_sub_1 = hf_hub_download(repo_id=repo_id, filename='sub_topic_1.joblib')
    model_sub_1 = joblib.load(model_sub_1)
    model_sub_min1 = hf_hub_download(repo_id=repo_id, filename='sub_topic_min1.joblib')
    model_sub_min1 = joblib.load(model_sub_min1)
    topic_model = model_utama
    sub_topic_model_min1 =  model_sub_min1
    sub_topic_model_1 = model_sub_1

    # Define helper functions for sub-topic classification
    def classify_sub_topic_min1(sub_topic_model, data):
        sub_topics, sub_probs = sub_topic_model.transform(data)
        if sub_topics[0] == 2:
            return 2
        else:
            return -1

    def classify_sub_topic_1(sub_topic_model, data):
        sub_topics, sub_probs = sub_topic_model.transform(data)
        return sub_topics[0]

    # Define function to get final topic name based on model predictions
    def get_final_topic(complaint):
        topics, probs = topic_model.transform([complaint])
        main_topic = topics[0]
        if main_topic == -1:
            sub_topic = classify_sub_topic_min1(sub_topic_model_min1, [complaint])
            if sub_topic == 2:
                return 'Kendala Install Aplikasi Client'
            else:
                return 'Outlier'
        elif main_topic == 1:
            sub_topic = classify_sub_topic_1(sub_topic_model_1, [complaint])
            if sub_topic == 0:
                return 'Kendala Akses Aplikasi/Authorized Error'
            elif sub_topic == 1:
                return 'Lupa Password/Reset Password'
            elif sub_topic == 2:
                return 'Kendala Username Salah'
        elif main_topic == 0:
            return 'Permintaan Akses ARO/RO'
        elif main_topic == 2:
            return 'Kesalahan Validasi Form'
        elif main_topic == 3:
            return 'Perpanjangan Waktu Pengumpulan'
        elif main_topic == 4:
            return 'Gagal Upload Pelaporan'
        elif main_topic == 5:
            return 'Kendala Opsi untuk Syariah'
        return 'Unknown'

    df['Topic_Name'] = df['Cleaned_Complaint'].apply(get_final_topic)

    # Hitung jumlah masing-masing topik
    topic_counts = df.groupby('Topic_Name').size().reset_index(name='Count')
    df['Topic_Name_Count'] = df['Topic_Name'].map(topic_counts.set_index('Topic_Name')['Count'])

    st.write("#### Distribusi Topik Final:")
    st.dataframe(topic_counts)

    # Tampilkan full teks Cleaned Complaint menggunakan iterrows (jika diinginkan)
    st.write("#### Full Text of Cleaned Complaint per Incident:")
    for idx, row in df.iterrows():
        st.write("=" * 135)
        st.write(f"**Incident Number**: {row['Incident Number']}")
        st.write(f"**Complaint**: {row['Cleaned_Complaint']}")
        st.write("=" * 135)
        st.write("")
else:
    st.info("Silakan upload file Excel atau CSV untuk memulai proses.")

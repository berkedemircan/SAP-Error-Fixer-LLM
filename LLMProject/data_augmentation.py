import json
import openai
import random
import time
import os
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = 'SAP_Logs.json'
OUTPUT_FILE = 'Users/berkedemircan/Documents/LLMProject/SAP_Logs_1500_gpt.json'
TARGET_COUNT = 1500

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

# Variant Augmentation
async def gpt_varyasyon(prompt, retries=3):
    for i in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sen bir SAP Data Services log varyasyon üreticisisin. Sana verilen log kaydını anlamını bozmadan, ama içeriğini (tablo adı, job adı, hata kodu, açıklama, tarih, sayı, vs.) değiştirerek yeni ve gerçekçi bir log olarak döndür. Sadece yeni logu JSON formatında döndür."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.9,
                n=1
            )
            return response.choices[0].message['content']
        except Exception as e:
            print(f"GPT hatası: {e}. {i+1}. deneme.")
            time.sleep(2)
    return None

def main():
    with open(INPUT_FILE, encoding='utf-8') as f:
        logs = json.load(f)
    new_logs = list(logs)
    print("GPT ile varyasyon üretimi başlıyor...")
    while len(new_logs) < TARGET_COUNT:
        base_entry = random.choice(logs)
        prompt = f"Aşağıdaki SAP Data Services log kaydını anlamını bozmadan, ama içeriğini (tablo adı, job adı, hata kodu, açıklama, tarih, sayı, vs.) değiştirerek yeni ve gerçekçi bir log olarak döndür. Sadece yeni logu JSON formatında döndür.\n\n{json.dumps(base_entry, ensure_ascii=False)}"
        
        new_log_str = gpt_varyasyon(prompt)
        if new_log_str:
            try:
                new_log = json.loads(new_log_str)
                new_logs.append(new_log)
                print(f"{len(new_logs)}/{TARGET_COUNT} log üretildi.")
            except Exception as e:
                print(f"JSON parse hatası: {e}")
        else:
            print("GPT'den yanıt alınamadı, tekrar denenecek.")
        time.sleep(1.5)  
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_logs, f, ensure_ascii=False, indent=4)
    print(f"{OUTPUT_FILE} dosyası oluşturuldu. Toplam kayıt: {len(new_logs)}")

if "_name_" == "_main_":
    main()
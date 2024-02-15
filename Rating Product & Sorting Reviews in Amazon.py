
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f'% x)



###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################
df = pd.read_csv(r"C:\Users\muhammet.guneri\Desktop\Rating Product&SortingReviewsinAmazon\amazon_review.csv")
df.columns
df.head()
df.groupby("asin").agg({"overall":"mean"}) #4.5875 ürünün genel ortalaması
df["overall"].mean() #aynı şey ürünün genel ortalaması

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
###################################################

df.head()
df.info() #değişkenlerimin cinslerine bakıyorum
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df.info()

df["day_diff"].describe().T #25% 281 - mean 437
df.loc[df["day_diff"] <= 300, "overall"].mean() #4.69
df.loc[(df["day_diff"] > 300) & (df["day_diff"] <= 500), "overall"].mean() #4.59
df.loc[(df["day_diff"] > 500) & (df["day_diff"] <= 750), "overall"].mean() #4.51
df.loc[df["day_diff"] > 750, "overall"].mean() #4.38

df.loc[df["day_diff"] <= 300, "overall"].mean() * 30/100 +\
df.loc[(df["day_diff"] > 300) & (df["day_diff"] <= 500), "overall"].mean() * 28/100 +\
df.loc[(df["day_diff"] > 500) & (df["day_diff"] <= 750), "overall"].mean() * 22/100 +\
df.loc[df["day_diff"] > 750, "overall"].mean() * 20/100

#4.5675
# örnek çözümde quantile kullanılarak yapabiliriz
def time_based_weighted_average(dataframe, w1=50, w2=25, w3=15, w4=10):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w4 / 100

###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################


###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

df.head()
# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

#Toplam oy sayısından (total_vote) yararlı oy sayısı (helpful_yes) çıkarılarak yararlı bulunmayan oy sayılarını (helpful_no) bulunuz.
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]
df.head()

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################
def score_pos_neg_diff (up,down):
    return up - down

def score_average_rating (up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound (up, down, confidence = 0.95):
    n = up + down
    if n == 0:
        return 0
    Z = st.norm.ppf(1-(1-confidence) / 2)
    phat= 1 * up / n
    return (phat + Z * Z /(2 * n)- Z* math.sqrt((phat*(1-phat)+Z*Z / (4*n)/n)) /(1+Z*Z/n))

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],x["helpful_no"]),axis=1)
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],x["helpful_no"]),axis=1)
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],x["helpful_no"]),axis=1)

df.head()


df["total_vote"].describe().T
df.sort_values("total_vote",ascending=False).head(20)
df[(df["total_vote"] > 0)].sort_values("total_vote",ascending=False).head(20)
df[(df["score_pos_neg_diff"] > 0)].sort_values("score_pos_neg_diff",ascending=False).head(20)
df[(df["wilson_lower_bound"] > 0)].sort_values("wilson_lower_bound",ascending=False).head(20)


# overall'ı da düşünebilirsin.

##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

# total_vote 'da eşik değeri veriyorum 0 verince wilsona göre sırala dediğimde total vote'u 1 olanları veriyor.
filtered_df = df[(df["total_vote"] > 8) & (df["score_pos_neg_diff"] > 0) & (df["wilson_lower_bound"] > 0)]


sorted_df = filtered_df.sort_values("wilson_lower_bound", ascending=False)

# İlk 20 kaydı göster
print(sorted_df.head(20))

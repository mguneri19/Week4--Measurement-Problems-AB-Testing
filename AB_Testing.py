#####################################################
# AB Testi ile Bidding Yöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler iç
# in Purchasemetriğine odaklanılmalıdır.




#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’inin ayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna Average Bidding uygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç


import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: "%.4f" %x)


#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.




#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################

# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test grubu verilerini ayrı değişkenlere atayınız.

df = pd.read_excel(r"C:\Users\muhammet.guneri\Desktop\ABTesti\ab_testing.xlsx")
df.head()
# Kontrol grubuna Maximum Bidding, test grubuna Average Bidding
maximum_bidding = pd.read_excel(r"C:\Users\muhammet.guneri\Desktop\ABTesti\ab_testing.xlsx", sheet_name="Control Group")

average_bidding = pd.read_excel(r"C:\Users\muhammet.guneri\Desktop\ABTesti\ab_testing.xlsx", sheet_name="Test Group")

maximum_bidding.head()
average_bidding.head()


# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.

maximum_bidding.describe().T # kontrol grubuna bakıyorum, mean ile median birbirine yakın, normal dağılım gibi.

average_bidding.describe().T # test grubuna bakıyorum. mean ile median birbirine yakın, normal dağılım gibi.

# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.

merge_bidding = pd.concat([maximum_bidding, average_bidding], axis=1) #alt alta control ve test gruplarını ekledim
merge_bidding.head()

merge_bidding.describe().T # tek bir seferde iki ayrı grubun istatistiki değerlerini inceleyebiliyorum.

#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.
# H0 : M1 = M2 ( Kontrol grup ile Test grupları ortalama purchase değerleri arasında farklılık yoktur.)
# H1 : M1 != M2 (Kontrol grup ile Test grupları ortalama purchase değerleri arasında farklılık vardır.)


# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz
maximum_bidding["Purchase"].mean() # 550.89
average_bidding["Purchase"].mean() # 582.10
# Kontrol grubunun kazanç ortalaması test grubuna göre daha düşük seviyede gözleniyor.

#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz

#normallik varsayımı
# H0 : Kontrol ve Test grupları ortalama purchase değerleri normal dağılımı sağlamaktadır.
# H1: Kontrol ve Test grupları ortalama purchase değerleri normal dağılımı sağlamamaktadır.

test_stat, pvalue = shapiro(maximum_bidding["Purchase"])
print("Test_Stat = %.4f, p-value = %.4f" % (test_stat, pvalue)) #p-value değeri = 0.58 > 0.05 H0: Rededilmez (kontrol grup)

test_stat, pvalue = shapiro(average_bidding["Purchase"])
print("Test_Stat = %.4f, p-value = %.4f" % (test_stat, pvalue)) #p-value değeri = 0.1541 > 0.05 H0: Rededilmez (test grup)

# her iki grupta da p-value değeri 0.05'ten büyük olduğu için varyans homojenliğine bakıyoruz.
# H0: rededilemez. Her iki değer de normal dağılımı sağlamaktadır.

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz
# H0 : Kontrol ve Test grupları ortalama purchase değerleri varyans homojenliği sağlamaktadır.
# H1 : Kontrol ve Test grupları ortalama purchase değerleri varyans homojenliği sağlamamaktadır.

test_stat, pvalue = levene(maximum_bidding["Purchase"],
                           average_bidding["Purchase"])
print("Test_Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value değeri 0.1083 > 0.05 olduğu için H0: rededilemez, varyans homojenliğini sağlamaktadır.
# her iki varsayım sağlandığı için parametrik  t-test kullanılır.

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.

test_stat, pvalue = ttest_ind(maximum_bidding["Purchase"],
                              average_bidding["Purchase"], equal_var=True) # her iki koşul sağlandığı için True diyoruz.

print("Test_Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value değeri 0.34 > 0.05 olduğu için hipotezimizdeki H0 değeri rededilemez.
# Yani, Kontrol grup ile Test grupları ortalama purchase değerleri arasında istatistiki olarak anlamlı bir farklılık yoktur.
# Test grubundaki purchase değerinin ortalamasının (582.10) Kontrol grubundakinden (550.89) yüksek olmasını şans eseri olduğunu söyleyebiliriz.

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.

# Hipotezler kurulduktan sonra normallik dağılımı ve varyans homojenliği varsayımları için testler kullanıldı.
# Normallik dağılımı için shapiro testi, varyans homojenliği için levene testleri kullanıldı.
# P-value değeri 0.05'ten büyük çıktığı için her iki varsayım sağlandığı için ttest kullanıldı.
# ttest sonucunda p value değeri 0.34 olduğu için hipotezimiz H0 rededilemez.



# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.

# Özetle, Kontrol grup ile Test grupları ortalama purchase değerleri arasında istatistiki anlamlı bir farklılık yoktur.
# Diğer değişkenler kullanılarak hipotez testleri yapılabilir. Bu sayede daha kapsamlı bir değerlendirme ortaya çıkabilir.
# Şirketin stratejisine göre purchase ortalaması yerine earning ortalaması için hipotez testi yapılarak kazanç arasında farklılık
# olup olmadığına bakılabilir.
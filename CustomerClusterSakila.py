from connectors.connector import Connector
import threadpoolctl
# ponytail: threadpoolctl đọc sai đường dẫn có dấu ("Học") làm KMeans lỗi DLL, ép dùng cách dò DLL dự phòng
def _noSnapshot(self, kernel_32): raise OSError
threadpoolctl.ThreadpoolController._snapshot_loaded_modules = _noSnapshot
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Thuộc tính đề xuất (mỗi dòng = 1 khách hàng):
# - TotalRentals       (rental)                  : tần suất thuê -> mức độ quan tâm chung
# - DistinctCategories (inventory->film->category): độ rộng sở thích thể loại phim
# - AvgRentalRate      (film)                    : thích phim giá cao (premium) hay phim rẻ
# - AvgDaysKept        (rental)                  : giữ bản sao (inventory) bao lâu -> mức chiếm dụng kho
SQL = """
select c.customer_id as CustomerId,
       concat(c.first_name, ' ', c.last_name) as Name,
       count(r.rental_id) as TotalRentals,
       count(distinct fc.category_id) as DistinctCategories,
       round(avg(f.rental_rate), 2) as AvgRentalRate,
       round(avg(datediff(r.return_date, r.rental_date)), 2) as AvgDaysKept
from customer c
join rental r on r.customer_id = c.customer_id
join inventory i on i.inventory_id = r.inventory_id
join film f on f.film_id = i.film_id
join film_category fc on fc.film_id = f.film_id
group by c.customer_id, Name
"""
FEATURES = ['TotalRentals', 'DistinctCategories', 'AvgRentalRate', 'AvgDaysKept']
RESULT = Path(__file__).resolve().parent / "result"
RESULT.mkdir(exist_ok=True)

conn = Connector(server="localhost", port=3306, database="sakila", username="root", password="@Obama123")
conn.connect()
df = conn.queryDataset(SQL)
conn.disConnect()
df[FEATURES] = df[FEATURES].astype(float)
print(df.describe())

# các thuộc tính khác đơn vị (lượt, thể loại, $, ngày) -> chuẩn hoá trước khi KMeans
X = StandardScaler().fit_transform(df[FEATURES])

def findElbow(ks, inertia):
    # điểm xa nhất so với đường thẳng nối điểm đầu - điểm cuối của đường inertia
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(inertia) - min(inertia)) / (max(inertia) - min(inertia))
    return ks[int(np.argmax((1 - x) - y))]

ks = list(range(1, 11))
inertia = [KMeans(n_clusters=k, init='k-means++', max_iter=500, random_state=42).fit(X).inertia_ for k in ks]
optimalK = findElbow(ks, inertia)
print("Số cụm tối ưu:", optimalK)
plt.figure(figsize=(10, 5))
plt.plot(ks, inertia, 'o-')
plt.axvline(optimalK, color='red', linestyle='--', label=f"Optimal k = {optimalK}")
plt.xlabel("Number of clusters"), plt.ylabel("Inertia"), plt.legend()
plt.savefig(RESULT / "elbow.png")
plt.show()

df['Cluster'] = KMeans(n_clusters=optimalK, init='k-means++', max_iter=500, random_state=42).fit_predict(X)
profile = df.groupby('Cluster')[FEATURES].mean().round(2)
profile['Count'] = df['Cluster'].value_counts().sort_index()
print(profile)

fig = px.scatter_3d(df, x='TotalRentals', y='AvgRentalRate', z='AvgDaysKept', color=df['Cluster'].astype(str),
                    hover_data=['Name', 'DistinctCategories'], title="Phân cụm khách hàng Sakila",
                    labels={'color': 'Cụm'})
fig.write_html(RESULT / "clusters_3d.html")
fig.show()

with pd.ExcelWriter(RESULT / "SakilaCustomerClusters.xlsx") as writer:
    profile.to_excel(writer, sheet_name="Profile")
    for k, g in df.groupby('Cluster'):
        g.to_excel(writer, sheet_name=f"Cluster {k}", index=False)

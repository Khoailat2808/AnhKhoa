from html import escape
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# bảng màu phân loại (thứ tự cố định, đã kiểm định cho người mù màu); dark mode đổi trong JS
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SYMBOLS = ['circle', 'square', 'diamond', 'triangle-up', 'cross', 'x', 'star', 'hexagon']
DIVERGING = [[0, '#1c5cab'], [0.5, '#f0efec'], [1, '#c43d3c']]
MUTED, GRID = '#898781', 'rgba(137,135,129,0.22)'

LABELS = {
    'TotalRentals': 'Số lượt thuê',
    'DistinctCategories': 'Số thể loại',
    'AvgRentalRate': 'Giá thuê TB ($)',
    'AvgDaysKept': 'Số ngày giữ TB',
}
SOURCES = {
    'TotalRentals': ('rental', 'Tần suất thuê – mức độ quan tâm chung tới cửa hàng'),
    'DistinctCategories': ('inventory → film → film_category', 'Độ rộng sở thích: xem đa thể loại hay chuyên một vài thể loại'),
    'AvgRentalRate': ('inventory → film', 'Khẩu vị giá: ưa phim premium hay phim giá rẻ'),
    'AvgDaysKept': ('rental (bản sao inventory)', 'Thời gian chiếm giữ bản sao – áp lực lên tồn kho'),
}
# (thuộc tính, dấu) -> (tên cụm, hành động kinh doanh)
PERSONAS = {
    ('TotalRentals', 1): ('Khách trung thành', 'Chương trình thành viên, tích điểm; ưu tiên giữ chân vì đóng góp nhiều lượt thuê nhất.'),
    ('TotalRentals', -1): ('Khách ít tương tác', 'Chiến dịch kích hoạt lại: mã giảm giá lượt thuê kế tiếp, gợi ý phim theo lịch sử.'),
    ('DistinctCategories', 1): ('Người khám phá', 'Gợi ý chéo thể loại, gói "thử thể loại mới", danh sách phim tuyển chọn.'),
    ('DistinctCategories', -1): ('Fan một thể loại', 'Bổ sung bản sao inventory cho thể loại yêu thích; thông báo khi có phim mới cùng thể loại.'),
    ('AvgRentalRate', 1): ('Ưa phim cao cấp', 'Ưu tiên phim mới/premium, tăng số bản sao phim giá cao tại cửa hàng của nhóm này.'),
    ('AvgRentalRate', -1): ('Săn phim giá rẻ', 'Combo giá rẻ, khuyến mãi theo gói; upsell nhẹ sang phim giá trung bình.'),
    ('AvgDaysKept', 1): ('Giữ đĩa lâu', 'Nhắc hạn trả, phí trễ hạn rõ ràng; cân nhắc gói thuê dài ngày để tận dụng hành vi.'),
    ('AvgDaysKept', -1): ('Trả đĩa nhanh', 'Đối tượng lý tưởng cho phim hot ít bản sao – kho xoay vòng nhanh; thưởng trả sớm.'),
}

CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;
--ring:rgba(11,11,11,.10);--accent:#2a78d6;--good:#006300;--bad:#d03b3b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--line:#2c2c2a;--ring:rgba(255,255,255,.10);--accent:#3987e5;--good:#0ca30c;--bad:#e66767}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#2c2c2a;
--ring:rgba(255,255,255,.10);--accent:#3987e5;--good:#0ca30c;--bad:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.6 "Inter","Segoe UI",system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:40px 16px 64px}
header h1{font-size:30px;line-height:1.2;margin:0 0 8px;letter-spacing:-.01em}
header p{color:var(--ink2);margin:0;max-width:760px}
h2{font-size:20px;margin:48px 0 4px}
h2 small{display:block;font-size:14px;font-weight:400;color:var(--ink2);margin-top:4px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:14px;padding:20px;margin-top:16px}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr))}
.grid>*{min-width:0;margin-top:0}
.grid.four{grid-template-columns:repeat(auto-fit,minmax(min(100%,250px),1fr))}
.kpis{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));margin-top:24px}
.kpi{background:var(--surface);border:1px solid var(--ring);border-radius:14px;padding:16px 18px}
.kpi b{display:block;font-size:28px;line-height:1.2;font-variant-numeric:tabular-nums}
.kpi span{color:var(--ink2);font-size:13px}
.kpi em{display:block;font-style:normal;color:var(--muted);font-size:12px;margin-top:2px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600;font-size:13px}
td.n,th.n{text-align:right}
.wrap{overflow-x:auto}
.sw{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:8px;vertical-align:baseline}
.tag{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line);color:var(--ink2);margin:2px 4px 2px 0}
.ok{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}
.persona h3{margin:0 0 2px;font-size:17px}
.persona .meta{color:var(--ink2);font-size:13px;margin-bottom:10px}
.persona p{margin:10px 0 0}
.note{color:var(--ink2);font-size:13px;margin:10px 0 0}
code{font-size:13px;background:var(--page);padding:1px 5px;border-radius:5px}
footer{color:var(--muted);font-size:13px;margin-top:48px}
"""

JS = """
const DARK={'#2a78d6':'#3987e5','#eb6834':'#d95926','#1baf7a':'#199e70','#eda100':'#c98500',
'#e87ba4':'#d55181','#008300':'#008300','#6250d6':'#9085e9','#e34948':'#e66767'};
const LIGHT=Object.fromEntries(Object.entries(DARK).map(([k,v])=>[v,k]));
function isDark(){const t=document.documentElement.dataset.theme;
 return t?t==='dark':matchMedia('(prefers-color-scheme: dark)').matches}
function swap(c,map){return typeof c==='string'?(map[c]||c):Array.isArray(c)?c.map(x=>swap(x,map)):c}
function theme(){const d=isDark(),map=d?DARK:LIGHT,ink=d?'#c3c2b7':'#52514e',
 mid=d?'#383835':'#f0efec',ring=d?'#1a1a19':'#fcfcfb';
 document.querySelectorAll('.js-plotly-plot').forEach(g=>{
  Plotly.relayout(g,{'font.color':ink});
  g.data.forEach((t,i)=>{const u={};
   if(t.marker&&t.marker.color)u['marker.color']=[swap(t.marker.color,map)];
   if(t.marker&&t.marker.line)u['marker.line.color']=[ring];
   if(t.line&&t.line.color)u['line.color']=[swap(t.line.color,map)];
   if(t.fillcolor)u.fillcolor=[t.fillcolor];
   if(t.type==='heatmap')u.colorscale=[[[0,'#1c5cab'],[.5,mid],[1,'#c43d3c']]];
   Plotly.restyle(g,u,[i])})})}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',theme);
new MutationObserver(theme).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
window.addEventListener('load',()=>{document.querySelectorAll('.js-plotly-plot').forEach(g=>Plotly.Plots.resize(g));theme()});
"""


def _style(fig, height=360):
    fig.update_layout(height=height, template='none', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      font=dict(family='Inter, Segoe UI, system-ui, sans-serif', size=13, color='#52514e'),
                      margin=dict(l=56, r=16, t=36, b=48), hoverlabel=dict(font_size=13),
                      legend=dict(orientation='h', y=1.08, x=0, bgcolor='rgba(0,0,0,0)'))
    fig.update_xaxes(automargin=True, gridcolor=GRID, zeroline=False, linecolor=MUTED, tickcolor=MUTED, title_font_color=MUTED)
    fig.update_yaxes(automargin=True, gridcolor=GRID, zeroline=False, linecolor=MUTED, tickcolor=MUTED, title_font_color=MUTED)
    return fig.to_html(full_html=False, include_plotlyjs=False, config={'displaylogo': False, 'responsive': True})


def _fmt_p(p):
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def export_report(df, X, features, ks, inertia, k, filename):
    clusters = sorted(df['Cluster'].unique())
    names = [f'Cụm {c}' for c in clusters]
    color = {c: COLORS[i] for i, c in enumerate(clusters)}
    n = len(df)

    # --- chất lượng phân cụm theo k ---
    sil = {kk: silhouette_score(X, _fit(X, kk)) for kk in ks if kk >= 2}
    labels = df['Cluster'].values
    sil_k, db_k, ch_k = silhouette_score(X, labels), davies_bouldin_score(X, labels), calinski_harabasz_score(X, labels)

    # --- kiểm định khác biệt giữa các cụm ---
    tests = []
    for f in features:
        groups = [df.loc[df['Cluster'] == c, f] for c in clusters]
        F, p = stats.f_oneway(*groups)
        H, pk = stats.kruskal(*groups)
        grand = df[f].mean()
        eta2 = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / ((df[f] - grand) ** 2).sum()
        tests.append((f, F, p, H, pk, eta2))

    # --- hồ sơ cụm: z-score tâm cụm trong không gian đã chuẩn hoá ---
    z = pd.DataFrame(X, columns=features).groupby(labels).mean()

    # ---------- biểu đồ ----------
    elbow = go.Figure(go.Scatter(x=ks, y=inertia, mode='lines+markers', line=dict(color=COLORS[0], width=2),
                                 marker=dict(size=8, color=COLORS[0]), hovertemplate='k = %{x}<br>Inertia = %{y:,.0f}<extra></extra>'))
    elbow.add_vline(x=k, line=dict(color=MUTED, dash='dash', width=1))
    elbow.add_annotation(x=k, y=max(inertia), text=f'k = {k}', showarrow=False, xanchor='left', xshift=6, font_color=MUTED)
    elbow.update_layout(xaxis_title='Số cụm k', yaxis_title='Inertia (WCSS)', showlegend=False)

    sk = list(sil)
    silfig = go.Figure(go.Bar(x=sk, y=[sil[s] for s in sk], marker=dict(color=[COLORS[0] if s == k else '#b7d3f6' for s in sk]),
                              hovertemplate='k = %{x}<br>Silhouette = %{y:.3f}<extra></extra>'))
    silfig.update_layout(xaxis_title='Số cụm k', yaxis_title='Silhouette TB', showlegend=False, bargap=0.35)

    counts = df['Cluster'].value_counts().sort_index()
    sizes = go.Figure(go.Bar(y=names[::-1], x=counts.values[::-1], orientation='h',
                             marker=dict(color=[color[c] for c in clusters][::-1]),
                             text=[f'{v} KH · {v / n:.0%}' for v in counts.values[::-1]], textposition='outside',
                             cliponaxis=False, hovertemplate='%{y}: %{x} khách hàng<extra></extra>'))
    sizes.update_layout(xaxis_title='Số khách hàng', showlegend=False, bargap=0.4, xaxis_range=[0, counts.max() * 1.3])

    heat = go.Figure(go.Heatmap(z=z.values, x=[LABELS[f] for f in features], y=names, colorscale=DIVERGING, zmid=0,
                                zmin=-max(1, abs(z.values).max()), zmax=max(1, abs(z.values).max()),
                                text=[[f'{v:+.2f}σ' for v in row] for row in z.values], texttemplate='%{text}',
                                xgap=2, ygap=2, colorbar=dict(title='z', thickness=10, outlinewidth=0),
                                hovertemplate='%{y} · %{x}<br>Lệch %{z:+.2f} độ lệch chuẩn so với TB<extra></extra>'))
    heat.update_yaxes(autorange='reversed', gridcolor='rgba(0,0,0,0)')
    heat.update_xaxes(gridcolor='rgba(0,0,0,0)')

    corr = df[features].corr()
    corrfig = go.Figure(go.Heatmap(z=corr.values, x=[LABELS[f] for f in features], y=[LABELS[f] for f in features],
                                   colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1, xgap=2, ygap=2,
                                   text=corr.round(2).values, texttemplate='%{text}',
                                   colorbar=dict(title='r', thickness=10, outlinewidth=0),
                                   hovertemplate='%{y} × %{x}<br>r = %{z:.2f}<extra></extra>'))
    corrfig.update_yaxes(autorange='reversed', gridcolor='rgba(0,0,0,0)')
    corrfig.update_xaxes(gridcolor='rgba(0,0,0,0)')

    boxes = make_subplots(rows=2, cols=2, subplot_titles=[LABELS[f] for f in features], vertical_spacing=0.16)
    for i, f in enumerate(features):
        for c, name in zip(clusters, names):
            boxes.add_trace(go.Box(y=df.loc[df['Cluster'] == c, f], name=name, marker=dict(color=color[c], size=4),
                                   line=dict(color=color[c], width=1.5), boxmean=True, legendgroup=name,
                                   showlegend=(i == 0), hovertemplate=f'{name}<br>{LABELS[f]}: %{{y}}<extra></extra>'),
                            row=i // 2 + 1, col=i % 2 + 1)
    boxes.update_xaxes(showticklabels=False)
    boxes.update_annotations(font_size=13)
    boxes.update_layout(legend_y=1.14, margin_t=70)

    pca = PCA(n_components=2).fit(X)
    P = pca.transform(X)
    ev = pca.explained_variance_ratio_
    scatter = go.Figure()
    for i, (c, name) in enumerate(zip(clusters, names)):
        m = labels == c
        scatter.add_trace(go.Scatter(x=P[m, 0], y=P[m, 1], mode='markers', name=name,
                                     marker=dict(color=color[c], symbol=SYMBOLS[i], size=8, opacity=0.8,
                                                 line=dict(color='#fcfcfb', width=1)),
                                     customdata=df.loc[m, ['Name'] + features].values,
                                     hovertemplate='<b>%{customdata[0]}</b><br>' + name + '<br>' +
                                     '<br>'.join(f'{LABELS[f]}: %{{customdata[{j + 1}]}}' for j, f in enumerate(features)) +
                                     '<extra></extra>'))
    scatter.update_layout(xaxis_title=f'PC1 ({ev[0]:.0%} phương sai)', yaxis_title=f'PC2 ({ev[1]:.0%} phương sai)')
    loadings = pd.DataFrame(pca.components_.T, index=features, columns=['PC1', 'PC2'])

    # ---------- bảng ----------
    feat_rows = ''.join(f'<tr><td><b>{LABELS[f]}</b><br><code>{f}</code></td><td>{escape(SOURCES[f][0])}</td>'
                        f'<td>{escape(SOURCES[f][1])}</td><td class="n">{df[f].mean():.2f}</td><td class="n">{df[f].std():.2f}</td>'
                        f'<td class="n">{df[f].min():g} – {df[f].max():g}</td></tr>' for f in features)

    test_rows = ''.join(
        f'<tr><td>{LABELS[f]}</td><td class="n">{F:.1f}</td><td class="n">{_fmt_p(p)}</td><td class="n">{H:.1f}</td>'
        f'<td class="n">{_fmt_p(pk)}</td><td class="n">{e:.2f}</td><td>{_effect(e)}</td>'
        f'<td>{"<span class=ok>✓ Khác biệt</span>" if p < 0.05 else "<span class=no>✗ Không</span>"}</td></tr>'
        for f, F, p, H, pk, e in tests)

    prof_head = ''.join(f'<th class="n">{LABELS[f]}</th>' for f in features)
    prof_rows = ''
    for c, name in zip(clusters, names):
        g = df[df['Cluster'] == c]
        cells = ''.join(f'<td class="n">{g[f].mean():.2f} <span style="color:var(--muted)">± {g[f].std():.2f}</span></td>' for f in features)
        prof_rows += f'<tr><td><span class="sw" style="background:{color[c]}"></span>{name}</td><td class="n">{len(g)}</td>{cells}</tr>'
    cells = ''.join(f'<td class="n"><b>{df[f].mean():.2f}</b></td>' for f in features)
    prof_rows += f'<tr><td><b>Toàn bộ</b></td><td class="n"><b>{n}</b></td>{cells}</tr>'

    load_rows = ''.join(f'<tr><td>{LABELS[f]}</td><td class="n">{r.PC1:+.2f}</td><td class="n">{r.PC2:+.2f}</td></tr>'
                        for f, r in loadings.iterrows())

    personas = ''
    for c, name in zip(clusters, names):
        row = z.loc[c].sort_values(key=abs, ascending=False)
        top = row.index[0]
        title, action = PERSONAS[(top, 1 if row.iloc[0] > 0 else -1)]
        chips = ''.join(f'<span class="tag">{LABELS[f]} {v:+.2f}σ</span>' for f, v in row.items())
        personas += (f'<div class="card persona" style="border-top:4px solid {color[c]}"><h3>{name} · {title}</h3>'
                     f'<div class="meta">{counts[c]} khách hàng ({counts[c] / n:.0%}) · đặc trưng nổi bật nhất: '
                     f'{LABELS[top]}</div>{chips}<p><b>Đề xuất:</b> {action}</p></div>')

    total = int(df['TotalRentals'].sum())
    best = max(sil, key=sil.get)
    sil_note = (f'k = {k} cũng có Silhouette cao nhất.' if best == k else
                f'k = {best} có Silhouette cao hơn một chút ({sil[best]:.3f} so với {sil_k:.3f}) nhưng chỉ chia thô; '
                f'k = {k} giữ độ tách biệt tương đương mà cho nhiều phân khúc hành động được hơn.')
    html = f"""<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Phân cụm khách hàng Sakila</title>
<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>
<style>{CSS}</style></head><body><main>
<header><h1>Phân cụm khách hàng theo mức độ quan tâm Film &amp; Inventory</h1>
<p>Cơ sở dữ liệu <code>sakila</code> (MySQL) · bảng customer, rental, inventory, film (+ film_category) ·
K-Means trên {len(features)} thuộc tính đã chuẩn hoá z-score.</p></header>

<div class="kpis">
<div class="kpi"><span>Khách hàng</span><b>{n:,}</b><em>có ít nhất 1 lượt thuê</em></div>
<div class="kpi"><span>Lượt thuê</span><b>{total:,}</b><em>TB {total / n:.1f} lượt / khách</em></div>
<div class="kpi"><span>Số cụm (k)</span><b>{k}</b><em>chọn bằng Elbow</em></div>
<div class="kpi"><span>Silhouette</span><b>{sil_k:.3f}</b><em>−1 → 1, càng cao càng tách biệt</em></div>
<div class="kpi"><span>Davies–Bouldin</span><b>{db_k:.2f}</b><em>càng thấp càng tốt</em></div>
<div class="kpi"><span>Calinski–Harabasz</span><b>{ch_k:,.0f}</b><em>càng cao càng tốt</em></div>
</div>

<h2>1. Thuộc tính đề xuất<small>Mỗi khách hàng là một quan sát; thuộc tính được tổng hợp từ lịch sử thuê.</small></h2>
<div class="card wrap"><table><thead><tr><th>Thuộc tính</th><th>Nguồn</th><th>Ý nghĩa</th><th class="n">TB</th><th class="n">Độ lệch chuẩn</th><th class="n">Khoảng</th></tr></thead>
<tbody>{feat_rows}</tbody></table>
<p class="note">Các thuộc tính khác đơn vị (lượt, thể loại, $, ngày) nên được chuẩn hoá z = (x − μ) / σ trước khi tính khoảng cách Euclid.</p></div>
<div class="card"><b>Tương quan giữa các thuộc tính (Pearson r)</b>{_style(corrfig, 340)}
<p class="note">Tương quan thấp giữa các thuộc tính → mỗi thuộc tính mang thông tin riêng, không dư thừa.</p></div>

<h2>2. Chọn số cụm<small>Elbow cho điểm gãy của WCSS; Silhouette kiểm tra chéo độ tách biệt.</small></h2>
<div class="grid"><div class="card"><b>Elbow – Inertia theo k</b>{_style(elbow, 320)}</div>
<div class="card"><b>Silhouette theo k</b>{_style(silfig, 320)}
<p class="note">Cột đậm là k được chọn. {sil_note} Silhouette quanh {sil_k:.2f} là điển hình cho dữ liệu hành vi liên tục, không có ranh giới tự nhiên rõ – các cụm nên được đọc như phân khúc, không phải nhóm tách rời.</p></div></div>

<h2>3. Quy mô & hồ sơ cụm</h2>
<div class="grid"><div class="card"><b>Số khách hàng mỗi cụm</b>{_style(sizes, 300)}</div>
<div class="card"><b>Tâm cụm (z-score)</b>{_style(heat, 300)}
<p class="note">Đỏ: cao hơn trung bình toàn bộ; xanh: thấp hơn. Đơn vị là độ lệch chuẩn (σ).</p></div></div>
<div class="card wrap"><b>Trung bình ± độ lệch chuẩn theo cụm</b><table><thead><tr><th>Cụm</th><th class="n">Số KH</th>{prof_head}</tr></thead>
<tbody>{prof_rows}</tbody></table></div>

<h2>4. Kiểm định thống kê<small>Các cụm có khác biệt thật sự trên từng thuộc tính? H₀: trung bình (trung vị) các cụm bằng nhau, α = 0.05.</small></h2>
<div class="card wrap"><table><thead><tr><th>Thuộc tính</th><th class="n">ANOVA F</th><th class="n">p</th><th class="n">Kruskal H</th><th class="n">p</th>
<th class="n">η²</th><th>Mức ảnh hưởng</th><th>Kết luận</th></tr></thead><tbody>{test_rows}</tbody></table>
<p class="note">η² = tỷ lệ phương sai của thuộc tính được giải thích bởi việc chia cụm (0.01 nhỏ · 0.06 trung bình · 0.14 lớn).
Kruskal–Wallis là kiểm định phi tham số, dùng đối chiếu khi phân phối không chuẩn.
Lưu ý: cụm được tạo từ chính các thuộc tính này nên p nhỏ là kỳ vọng – η² mới cho biết thuộc tính nào <i>chi phối</i> việc phân cụm.</p></div>
<div class="card"><b>Phân phối từng thuộc tính theo cụm</b>{_style(boxes, 620)}
<p class="note">Hộp: tứ phân vị Q1–Q3, vạch liền: trung vị, vạch đứt: trung bình.</p></div>

<h2>5. Không gian cụm (PCA)<small>Chiếu {len(features)} chiều xuống 2 thành phần chính, giữ {ev.sum():.0%} tổng phương sai.</small></h2>
<div class="card">{_style(scatter, 520)}</div>
<div class="card wrap"><b>Trọng số (loadings)</b><table><thead><tr><th>Thuộc tính</th><th class="n">PC1</th><th class="n">PC2</th></tr></thead>
<tbody>{load_rows}</tbody></table><p class="note">Trị tuyệt đối lớn → thuộc tính đóng góp nhiều vào trục đó.</p></div>

<h2>6. Diễn giải & đề xuất kinh doanh<small>Tên cụm đặt theo thuộc tính lệch xa trung bình nhất (|z| lớn nhất).</small></h2>
<div class="grid four">{personas}</div>

<footer>Sinh tự động bởi <code>sakila/CustomerClusterSakila.py</code> · danh sách khách hàng từng cụm: <code>SakilaCustomerClusters.xlsx</code></footer>
</main><script>{JS}</script></body></html>"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Đã xuất báo cáo HTML: {filename}')


def _fit(X, k):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, init='k-means++', max_iter=500, random_state=42).fit_predict(X)


def _effect(e):
    return 'Lớn' if e >= 0.14 else 'Trung bình' if e >= 0.06 else 'Nhỏ'

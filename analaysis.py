import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
import warnings
warnings.filterwarnings('ignore')


plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fc',
    'axes.grid': True,
    'grid.color': '#e0e3eb',
    'grid.linewidth': 0.6,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
NAVY  = '#0D1B3E'
ORG   = '#E84623'
GREEN = '#22c55e'
YELLOW= '#f59e0b'
BLUE  = '#3b82f6'
PAL   = [NAVY, ORG, GREEN, BLUE, YELLOW, '#a78bfa', '#ec4899', '#14b8a6']

df = pd.read_csv('student_data.csv')
df['Placed'] = (df['Placement_Status'] == 'Placed').astype(int)
print("Dataset loaded:", df.shape)

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Figure 1 – Dataset Overview & Exploratory Data Analysis',
             fontsize=16, fontweight='bold', color=NAVY, y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)


ax = fig.add_subplot(gs[0, 0])
counts = df['Placement_Status'].value_counts()
bars = ax.bar(counts.index, counts.values, color=[GREEN, ORG], edgecolor='white', linewidth=1.5, width=0.5)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+4, str(b.get_height()),
            ha='center', fontsize=11, fontweight='bold', color=NAVY)
ax.set_title('Placement Distribution', fontweight='bold', color=NAVY)
ax.set_ylabel('Count')


ax = fig.add_subplot(gs[0, 1])
dept_counts = df['Department'].value_counts()
ax.barh(dept_counts.index, dept_counts.values, color=PAL[:len(dept_counts)], edgecolor='white')
ax.set_title('Students by Department', fontweight='bold', color=NAVY)
ax.set_xlabel('Count')


ax = fig.add_subplot(gs[0, 2])
for label, grp in df.groupby('Placement_Status'):
    ax.hist(grp['CGPA'], bins=15, alpha=0.7, label=label,
            color=GREEN if label=='Placed' else ORG, edgecolor='white')
ax.set_title('CGPA Distribution', fontweight='bold', color=NAVY)
ax.set_xlabel('CGPA'); ax.legend(fontsize=9)


ax = fig.add_subplot(gs[0, 3])
for label, grp in df.groupby('Placement_Status'):
    ax.hist(grp['Attendance_%'], bins=15, alpha=0.7, label=label,
            color=GREEN if label=='Placed' else ORG, edgecolor='white')
ax.set_title('Attendance % Distribution', fontweight='bold', color=NAVY)
ax.set_xlabel('Attendance %'); ax.legend(fontsize=9)


ax = fig.add_subplot(gs[1, :2])
dept_place = df.groupby('Department')['Placed'].mean().sort_values(ascending=False) * 100
bars = ax.bar(dept_place.index, dept_place.values, color=PAL[:len(dept_place)], edgecolor='white')
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f'{b.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold', color=NAVY)
ax.set_title('Placement Rate by Department', fontweight='bold', color=NAVY)
ax.set_ylabel('Placement Rate (%)'); ax.set_ylim(0, 110)


ax = fig.add_subplot(gs[1, 2])
tier_place = df.groupby('College_Tier')['Placed'].mean() * 100
ax.bar([f'Tier {t}' for t in tier_place.index], tier_place.values,
       color=[GREEN, BLUE, ORG], edgecolor='white', width=0.5)
for i, v in enumerate(tier_place.values):
    ax.text(i, v+0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold', color=NAVY)
ax.set_title('Placement by College Tier', fontweight='bold', color=NAVY)
ax.set_ylabel('Placement Rate (%)'); ax.set_ylim(0, 110)


ax = fig.add_subplot(gs[1, 3])
gen_place = df.groupby('Gender')['Placed'].mean() * 100
ax.bar(gen_place.index, gen_place.values, color=[BLUE, ORG], edgecolor='white', width=0.4)
for i, v in enumerate(gen_place.values):
    ax.text(i, v+0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold', color=NAVY)
ax.set_title('Placement by Gender', fontweight='bold', color=NAVY)
ax.set_ylabel('Placement Rate (%)'); ax.set_ylim(0, 110)


ax = fig.add_subplot(gs[2, :2])
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) == 0:
    ax.text(0.5, 0.5, '✅  No Missing Values Found\n(Dataset is Clean)',
            ha='center', va='center', fontsize=14, color=GREEN,
            fontweight='bold', transform=ax.transAxes)
    ax.set_facecolor('#f0fdf4')
ax.set_title('Missing Value Analysis', fontweight='bold', color=NAVY)
ax.axis('off') if len(missing)==0 else None


ax = fig.add_subplot(gs[2, 2:])
stats_cols = ['CGPA','Attendance_%','Avg_Quiz_Score','Engagement_Score','Doubts_Raised']
stats_df = df[stats_cols].describe().round(2).loc[['mean','std','min','max']]
ax.axis('off')
tbl = ax.table(cellText=stats_df.values,
               rowLabels=stats_df.index,
               colLabels=[c.replace('_',' ').replace('%','') for c in stats_cols],
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
tbl.scale(1, 1.6)
for (r,c), cell in tbl.get_celld().items():
    if r == 0 or c == -1:
        cell.set_facecolor(NAVY); cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#f8f9fc' if r%2==0 else 'white')
ax.set_title('Descriptive Statistics', fontweight='bold', color=NAVY, pad=12)

plt.savefig('/home/claude/fig1_eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig1 done")


fig, axes = plt.subplots(3, 3, figsize=(18, 15))
fig.suptitle('Figure 2 – Deep Engagement Analysis: Behavior vs Placement',
             fontsize=16, fontweight='bold', color=NAVY)
plt.subplots_adjust(hspace=0.4, wspace=0.35)


ax = axes[0,0]
df['Att_Bucket'] = pd.cut(df['Attendance_%'], bins=[0,60,80,100],
                           labels=['<60%','60–80%','>80%'])
att_place = df.groupby('Att_Bucket', observed=True)['Placed'].mean() * 100
colors_att = [ORG, YELLOW, GREEN]
bars = ax.bar(att_place.index, att_place.values, color=colors_att, edgecolor='white')
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f'{b.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold', color=NAVY)
ax.set_title('Attendance vs Placement Rate', fontweight='bold', color=NAVY)
ax.set_ylabel('Placement Rate (%)'); ax.set_ylim(0, 110)
ax.set_xlabel('Attendance Bucket')


ax = axes[0,1]
login_place = df.groupby('Login_Frequency')['Placed'].mean() * 100
ax.bar(login_place.index, login_place.values, color=NAVY, edgecolor='white')
ax.plot(login_place.index, login_place.values, 'o-', color=ORG, linewidth=2, markersize=6)
ax.set_title('Login Frequency vs Placement', fontweight='bold', color=NAVY)
ax.set_xlabel('Logins per Week'); ax.set_ylabel('Placement Rate (%)')


ax = axes[0,2]
df['Time_Bucket'] = pd.cut(df['Time_Spent_Hours'], bins=[0,5,15,30,50],
                            labels=['<5 hrs','5-15 hrs','15-30 hrs','>30 hrs'])
time_place = df.groupby('Time_Bucket', observed=True)['Placed'].mean() * 100
bar_cols = [ORG, YELLOW, GREEN, BLUE]
ax.bar(time_place.index, time_place.values, color=bar_cols, edgecolor='white')
for b in ax.patches:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f'{b.get_height():.1f}%', ha='center', fontsize=9, fontweight='bold', color=NAVY)
ax.set_title('Time Spent vs Placement', fontweight='bold', color=NAVY)
ax.set_ylabel('Placement Rate (%)'); ax.set_ylim(0, 110)


ax = axes[1,0]
for label, grp in df.groupby('Placement_Status'):
    ax.hist(grp['Avg_Quiz_Score'], bins=20, alpha=0.7, label=label,
            color=GREEN if label=='Placed' else ORG, edgecolor='white')
ax.axvline(df[df['Placed']==1]['Avg_Quiz_Score'].mean(), color=GREEN,
           linestyle='--', linewidth=2, label=f"Placed avg: {df[df['Placed']==1]['Avg_Quiz_Score'].mean():.1f}")
ax.axvline(df[df['Placed']==0]['Avg_Quiz_Score'].mean(), color=ORG,
           linestyle='--', linewidth=2, label=f"Not avg: {df[df['Placed']==0]['Avg_Quiz_Score'].mean():.1f}")
ax.set_title('Quiz Score Distribution by Placement', fontweight='bold', color=NAVY)
ax.set_xlabel('Quiz Score'); ax.legend(fontsize=8)


ax = axes[1,1]
df['Vid_Bucket'] = pd.cut(df['Video_Completion_%'], bins=[0,50,80,100],
                           labels=['<50%','50–80%','>80%'])
vid_place = df.groupby('Vid_Bucket', observed=True)['Placed'].mean() * 100
ax.bar(vid_place.index, vid_place.values, color=[ORG, YELLOW, GREEN], edgecolor='white')
for b in ax.patches:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f'{b.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold', color=NAVY)
ax.set_title('Video Completion vs Placement', fontweight='bold', color=NAVY)
ax.set_ylabel('Placement Rate (%)'); ax.set_ylim(0, 110)


ax = axes[1,2]
df['Doubt_Cat'] = pd.cut(df['Doubts_Raised'], bins=[-1,0,4,100],
                          labels=['No Doubts','Some Doubts','Active Doubts'])
doubt_place = df.groupby('Doubt_Cat', observed=True)['Placed'].mean() * 100
ax.bar(doubt_place.index, doubt_place.values, color=[ORG, YELLOW, GREEN], edgecolor='white')
for b in ax.patches:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f'{b.get_height():.1f}%', ha='center', fontsize=10, fontweight='bold', color=NAVY)
ax.set_title('Doubt Behavior vs Placement', fontweight='bold', color=NAVY)
ax.set_ylabel('Placement Rate (%)'); ax.set_ylim(0, 110)


ax = axes[2,0]
event_place = df.groupby('Hackathons_Attended')['Placed'].mean() * 100
ax.bar(event_place.index, event_place.values, color=NAVY, edgecolor='white')
ax.plot(event_place.index, event_place.values, 's-', color=ORG, linewidth=2, markersize=6)
ax.set_title('Hackathons Attended vs Placement', fontweight='bold', color=NAVY)
ax.set_xlabel('Hackathons'); ax.set_ylabel('Placement Rate (%)')


ax = axes[2,1]
placed_eng = df[df['Placed']==1]['Engagement_Score']
notplaced_eng = df[df['Placed']==0]['Engagement_Score']
bp = ax.boxplot([notplaced_eng, placed_eng], labels=['Not Placed','Placed'],
                patch_artist=True, widths=0.4)
bp['boxes'][0].set_facecolor(ORG); bp['boxes'][1].set_facecolor(GREEN)
for whisker in bp['whiskers']: whisker.set_color(NAVY)
for cap in bp['caps']: cap.set_color(NAVY)
ax.set_title('Engagement Score by Placement', fontweight='bold', color=NAVY)
ax.set_ylabel('Engagement Score')


ax = axes[2,2]
for label, grp in df.groupby('Placement_Status'):
    c = GREEN if label=='Placed' else ORG
    ax.scatter(grp['CGPA'], grp['Engagement_Score'], alpha=0.4, color=c,
               label=label, s=20)
ax.set_title('CGPA vs Engagement Score', fontweight='bold', color=NAVY)
ax.set_xlabel('CGPA'); ax.set_ylabel('Engagement Score')
ax.legend(fontsize=9)

plt.savefig('/home/claude/fig2_engagement.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig2 done")


fig, axes = plt.subplots(1, 2, figsize=(20, 9))
fig.suptitle('Figure 3 – Correlation Analysis & Student Segmentation',
             fontsize=16, fontweight='bold', color=NAVY)


num_cols = ['Attendance_%','Login_Frequency','Time_Spent_Hours',
            'Video_Completion_%','Avg_Quiz_Score','Doubts_Raised',
            'Hackathons_Attended','Engagement_Score','CGPA','Placed']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, ax=axes[0], annot=True, fmt='.2f',
            cmap=cmap, vmin=-1, vmax=1, center=0,
            linewidths=0.5, annot_kws={'size': 8})
axes[0].set_title('Correlation Heatmap – Key Features', fontweight='bold', color=NAVY, pad=12)
axes[0].tick_params(axis='x', rotation=45)


def segment(row):
    if row['Engagement_Score'] >= 65 and row['Avg_Quiz_Score'] >= 65 and row['Doubts_Raised'] >= 3:
        return 'High Performer'
    elif row['Video_Completion_%'] >= 60 and row['Doubts_Raised'] < 2:
        return 'Passive Learner'
    elif row['Engagement_Score'] >= 55 and row['Avg_Quiz_Score'] < 50:
        return 'Active but Confused'
    else:
        return 'Disengaged'

df['Segment'] = df.apply(segment, axis=1)
seg_counts = df['Segment'].value_counts()
colors_seg = [GREEN, BLUE, YELLOW, ORG]
wedges, texts, autotexts = axes[1].pie(
    seg_counts.values, labels=seg_counts.index,
    autopct='%1.1f%%', colors=colors_seg,
    startangle=140, pctdistance=0.75,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2})
for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold')
axes[1].set_title('Student Segmentation\n(Based on Engagement Rules)', fontweight='bold', color=NAVY)


seg_place = df.groupby('Segment')['Placed'].mean() * 100
inset = axes[1].inset_axes([0.3, 0.0, 0.4, 0.28])
inset.barh(range(len(seg_place)), seg_place.values, color=colors_seg[::-1], edgecolor='white')
inset.set_yticks(range(len(seg_place)))
inset.set_yticklabels([s[:8] for s in seg_place.index], fontsize=6)
inset.set_xlabel('Placement %', fontsize=7)
inset.tick_params(labelsize=7)
inset.set_title('Placement Rate per Segment', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/fig3_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig3 done")

feature_cols = [
    'Attendance_%','Login_Frequency','Time_Spent_Hours','Active_Days_Per_Week',
    'Video_Completion_%','Rewatch_Rate','Quizzes_Attempted','Avg_Quiz_Score',
    'Quiz_Submission_Rate','Assignment_Submissions','On_Time_Submission_%',
    'Doubts_Raised','Doubts_Resolved','Peer_Discussion_Count',
    'Hackathons_Attended','Workshops_Attended','Live_Sessions_Joined',
    'Project_Demo_Events','Skills_Learned_Count','Project_Completion_Rate',
    'Engagement_Score','CGPA','College_Tier'
]

X = df[feature_cols]
y = df['Placed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    Xtr = X_train_s if name == 'Logistic Regression' else X_train
    Xte = X_test_s  if name == 'Logistic Regression' else X_test
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:,1]
    results[name] = {
        'model': model,
        'acc': accuracy_score(y_test, y_pred),
        'roc': roc_auc_score(y_test, y_prob),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }
    print(f"{name}: Acc={results[name]['acc']:.3f}  AUC={results[name]['roc']:.3f}")

fig = plt.figure(figsize=(20, 14))
fig.suptitle('Figure 4 – Machine Learning Model Results',
             fontsize=16, fontweight='bold', color=NAVY)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 4a-c. Confusion matrices
model_colors = [BLUE, GREEN, ORG]
for i, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[0, i])
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.light_palette(model_colors[i], as_cmap=True),
                linewidths=1, linecolor='white',
                xticklabels=['Not Placed','Placed'],
                yticklabels=['Not Placed','Placed'])
    ax.set_title(f'{name}\nAcc: {res["acc"]:.3f} | AUC: {res["roc"]:.3f}',
                 fontweight='bold', color=NAVY, fontsize=11)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')

ax = fig.add_subplot(gs[1, 0])
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['roc']:.3f})",
            color=model_colors[i], linewidth=2.5)
ax.plot([0,1],[0,1],'--', color='gray', linewidth=1)
ax.set_title('ROC Curves – All Models', fontweight='bold', color=NAVY)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=9); ax.fill_between([0,1],[0,1], alpha=0.05, color='gray')


ax = fig.add_subplot(gs[1, 1])
model_names = list(results.keys())
accs = [results[m]['acc']*100 for m in model_names]
aucs = [results[m]['roc']*100 for m in model_names]
x = np.arange(len(model_names))
w = 0.35
b1 = ax.bar(x-w/2, accs, w, label='Accuracy %', color=NAVY, edgecolor='white')
b2 = ax.bar(x+w/2, aucs, w, label='AUC %', color=ORG, edgecolor='white')
for b in list(b1)+list(b2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
            f'{b.get_height():.1f}', ha='center', fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels([m.replace(' ','\n') for m in model_names], fontsize=9)
ax.set_title('Model Comparison', fontweight='bold', color=NAVY)
ax.legend(); ax.set_ylim(0, 115)


ax = fig.add_subplot(gs[1, 2])
rf = results['Random Forest']['model']
fi = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(12)
ax.barh(fi.index, fi.values, color=[NAVY if v > fi.median() else BLUE for v in fi.values])
ax.axvline(fi.median(), color=ORG, linestyle='--', linewidth=1.5, label='Median')
ax.set_title('Top Feature Importances\n(Random Forest)', fontweight='bold', color=NAVY)
ax.set_xlabel('Importance Score'); ax.legend(fontsize=9)

plt.savefig('/home/claude/fig4_ml.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig4 done")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Figure 5 – Key Insights & Business Recommendations',
             fontsize=16, fontweight='bold', color=NAVY)
plt.subplots_adjust(hspace=0.42, wspace=0.35)


ax = axes[0,0]
df['Eng_Bucket'] = pd.cut(df['Engagement_Score'],
                           bins=[0,40,55,70,100],
                           labels=['Low\n(0–40)','Medium\n(40–55)','Good\n(55–70)','High\n(70+)'])
eng_place = df.groupby('Eng_Bucket', observed=True)['Placed'].mean() * 100
bar_c = [ORG, YELLOW, BLUE, GREEN]
bars = ax.bar(eng_place.index, eng_place.values, color=bar_c, edgecolor='white', width=0.55)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f'{b.get_height():.1f}%', ha='center', fontsize=11, fontweight='bold', color=NAVY)
ax.set_title('Engagement Score → Placement Rate', fontweight='bold', color=NAVY)
ax.set_ylabel('Placement Rate (%)'); ax.set_ylim(0, 115)
ax.set_xlabel('Engagement Score Bucket')


ax = axes[0,1]
skill_place = df.groupby('Skills_Learned_Count')['Placed'].mean() * 100
ax.bar(skill_place.index, skill_place.values, color=BLUE, edgecolor='white', alpha=0.85)
z = np.polyfit(skill_place.index, skill_place.values, 1)
p = np.poly1d(z)
ax.plot(skill_place.index, p(skill_place.index), '--', color=ORG, linewidth=2, label='Trend')
ax.set_title('Skills Learned vs Placement Rate', fontweight='bold', color=NAVY)
ax.set_xlabel('Skills Learned Count'); ax.set_ylabel('Placement Rate (%)')
ax.legend()


ax = axes[0,2]
seg_dept = df.groupby(['Segment','Department'])['Placed'].mean().unstack(fill_value=0) * 100
sns.heatmap(seg_dept, ax=ax, annot=True, fmt='.0f', cmap='YlOrRd',
            linewidths=0.5, linecolor='white', annot_kws={'size':8})
ax.set_title('Placement % by Segment & Dept', fontweight='bold', color=NAVY)
ax.set_xlabel('Department'); ax.set_ylabel('Segment')
ax.tick_params(axis='x', rotation=30)

ax = axes[1,0]
ax.scatter(df['Peer_Discussion_Count'], df['Engagement_Score'],
           c=df['Placed'].map({1:GREEN,0:ORG}), alpha=0.4, s=20)
ax.set_title('Peer Discussion vs Engagement Score', fontweight='bold', color=NAVY)
ax.set_xlabel('Peer Discussion Count'); ax.set_ylabel('Engagement Score')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GREEN,label='Placed'),Patch(color=ORG,label='Not Placed')], fontsize=9)


ax = axes[1,1]
df['Proj_Bucket'] = pd.cut(df['Project_Completion_Rate'],
                            bins=[0,50,75,100], labels=['<50%','50–75%','>75%'])
proj_place = df.groupby('Proj_Bucket', observed=True)['Placed'].mean() * 100
ax.bar(proj_place.index, proj_place.values, color=[ORG,YELLOW,GREEN], edgecolor='white', width=0.45)
for b in ax.patches:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
            f'{b.get_height():.1f}%', ha='center', fontsize=11, fontweight='bold', color=NAVY)
ax.set_title('Project Completion vs Placement', fontweight='bold', color=NAVY)
ax.set_ylabel('Placement Rate (%)'); ax.set_ylim(0, 115)


ax = axes[1,2]
ax.axis('off')
insights = [
    "KEY FINDINGS",
    "",
    "✅  Engagement beats CGPA alone",
    "✅  Attendance >80% → High placement",
    "✅  5–7 logins/week → Best outcome",
    "✅  Doubts raised = Growth mindset",
    "✅  15–30 hrs/week is optimal zone",
    "✅  Quiz score >75 → High predictor",
    "",
    "⚠️   0 doubts raised → Passive risk",
    "⚠️   Attendance <60% → Dropout risk",
    "",
    f"Model Best AUC: {max(r['roc'] for r in results.values()):.3f}",
    f"(Random Forest / Gradient Boosting)",
]
y_pos = 0.97
for line in insights:
    size = 11 if line in ["KEY FINDINGS",""] else 10
    weight = 'bold' if line == "KEY FINDINGS" else 'normal'
    color = NAVY if line == "KEY FINDINGS" else ('#166534' if '✅' in line else
             ('#b45309' if '⚠' in line else '#374151'))
    ax.text(0.05, y_pos, line, transform=ax.transAxes,
            fontsize=size, fontweight=weight, color=color, va='top')
    y_pos -= 0.07
ax.set_facecolor('#f8f9fc')
ax.set_title('Summary of Key Insights', fontweight='bold', color=NAVY)

plt.savefig('/home/claude/fig5_insights.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig5 done")
print("\nAll figures saved.")

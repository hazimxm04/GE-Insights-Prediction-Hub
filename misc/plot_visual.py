#heatmap: see pattern coalition
coalition_color = {
    "Harapan": "#FF0000",      # bright red
    "BN": "#0000FF",      # bright blue
    "PAS": "#90ee90",     # green
    "BEBAS": "#A9A9A9"    # grey
}

#convert hex to rgb format
coalition_rgb = {k: np.array(mcolors.to_rgb(v)) for k, v in coalition_color.items()}

#winners filter function
winners = df_parliment[df_parliment["Status"] == "Win"]
#to count coalition win
state_coalition_win= winners.groupby(["Region", "Coalition"]).size().unstack(fill_value = 0)

#normalization
state_pct_win = state_coalition_win.div(state_coalition_win.sum(axis =1), axis=0)

#combine by blend colors
def clr_combine(row):
    color = np.zeros(3)
    for coalition, value in row.items():
        if coalition in coalition_rgb:
            color += value * coalition_rgb[coalition]
    return tuple(np.clip(color, 0, 1))

#contain list (r,g,b) between 0-1 (% based)
state_colors = state_pct_win.apply(clr_combine, axis =1)

state_coalition_win.plot(
    kind = "barh",
    stacked = True,
    color=[coalition_color.get(c, "black") for c in state_coalition_win.columns],
    figsize=(10, 8)
)
df_parliment.to_csv("ParlimentGE14_V2.csv", index=False)

#plt.xlabel("Number Seats Won")
#plt.ylabel("State")
#plt.title("Coalition Wins by State")
#plt.tight_layout()
#plt.show()


#malaysia map plot
gdf = gpd.read_file("malaysia_states.json")
# Check the column naming for state names
print(gdf.head()) 

gdf["color"] = gdf["name"].apply(lambda s: clr_combine(state_pct_win.loc[s]))

fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(color=gdf["color"], ax=ax, edgecolor="black")
plt.title("GE14 Coalition Dominance by State")
plt.show()


#df_parliment.to_csv("ge14_visualization.csv", index=False)

#print(df_parliment["Pekerjaan"].unique())
#print(df_parliment.tail())
#print(df_parliment.info())

print(df_parliment.columns)
print(df_state.columns)
print(df_votes.columns)

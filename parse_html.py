import codecs
from bs4 import BeautifulSoup
import pandas as pd
from unidecode import unidecode

NAME_IDX = 0
STATE_IDX = 1
CHAMBER_IDX = 2
LEAN_IDX = 3
STATUS_IDX = 4
SEX_IDX = 5
FONT_IDX = 6
COLOR_IDX = 7


# create dataframe
col_names =  ['Name', 'Party', 'Chamber', 'State', 'District', 'Sex', 'Font','Color','Status','Lean']
my_df  = pd.DataFrame(columns = col_names)

f=codecs.open("raw_html.html", 'r', 'utf-8')
document= BeautifulSoup(f.read())

cards = document.body.find_all('div', class_='candidate-card-text')

# loop through cards
for i in range(len(cards)):
    cur_card = cards[i].find_all('li')
    name = unidecode(cur_card[NAME_IDX].text.split(' (')[0])
    party = cur_card[NAME_IDX].text.split(' (')[1].replace(')','')
    chamber = cur_card[CHAMBER_IDX].text
    if chamber == 'Senate':
        state = cur_card[STATE_IDX].text
        district = None
    else:
        state = cur_card[STATE_IDX].text.split('-')[0]
        district = cur_card[STATE_IDX].text.split('-')[1]
    sex = cur_card[SEX_IDX].text
    font = cur_card[FONT_IDX].text
    color = cur_card[COLOR_IDX].text
    status = cur_card[STATUS_IDX].text
    lean = cur_card[LEAN_IDX].text

    # append to dataframe
    cur_row = {'Name':name,
               'Party':party,
               'Chamber':chamber,
               'State':state,
               'District':district,
               'Sex':sex,
               'Font':font,
               'Color':color,
               'Status':status,
               'Lean':lean}
    my_df.loc[len(my_df)] = cur_row 

my_df.to_pickle("candidate-df.pb")

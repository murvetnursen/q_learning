
import numpy as np
import random
from matplotlib import pyplot 
from matplotlib import colors
import random
from tkinter import *

from numpy.lib.function_base import append

root = Tk()
root.title("REINFORECEMENT")
root.geometry("400x400")

def input():
    global i_row 
    global i_column 
    global t_row
    global t_column
    i_row = int(veri.get())
    i_column = int(veri2.get())
    t_row = int(veri3.get())
    t_column = int(veri4.get())
    
 
veri = Entry()
veri.pack()
label = Label(text = "girdi satır")
label.pack()
veri2 =Entry()
veri2.pack()
label2 = Label(text="girdi sütun")
label2.pack()
veri3 = Entry()
veri3.pack()
label3 = Label(text="hedef satır")
label3.pack()
veri4 =Entry()
veri4.pack()
label4 = Label(text="hedef sütun")
label4.pack()

button = Button(text="Start",command = input)
button.pack()

root.mainloop()


R = [[1 for i in range(20)] for j in range(6)]
W = [[0 for i in range(20)] for j in range(14)]
data = np.concatenate((R, W), axis = 0)
data = data.ravel()
random.shuffle(data)
data= data.reshape(20,20)

environment_rows = 20
environment_columns = 20


q_values = np.zeros((environment_rows, environment_columns, 8))
actions = ['up','right','down','left','uprcross','uplcross','downrcross','downlcross']

rewards = np.full((environment_rows, environment_columns), -1.)



for i in range(20):
  for j in range(20):
    if data[i][j] == 0:
      rewards[i,j] = -1.
    elif data[i][j] == 1:
      rewards[i][j] = -100.
     

rewards[i_row,i_column]=-1.
rewards[t_row,t_column] = 100.
data[i_row,i_column]=2
data[t_row,t_column]=2
for row in rewards:
  print(row)

dosya=open("engel.txt","w") 

for i in range(20):
  for j in range(20):
    if rewards[i,j]==-100.:
      icerik_K= "(" +str(i) +","+ str(j)+ ")->K   " 
      dosya.write(icerik_K)
    elif rewards[i,j]==-1.:
      icerik_B= "(" +str(i) +","+ str(j)+ ")->B   " 
      dosya.write(icerik_B)
    else:
      icerik_T= "(" +str(i) +","+ str(j)+ ")->T   " 
      dosya.write(icerik_T)

  dosya.write("\n")

def is_terminal_state(current_row_index, current_column_index):
  
  if rewards[current_row_index, current_column_index] == -1.:
    return False
  else:
    return True


def get_starting_location():

  current_row_index = np.random.randint(environment_rows)
  current_column_index = np.random.randint(environment_columns)
 
  while is_terminal_state(current_row_index, current_column_index):
    current_row_index = np.random.randint(environment_rows)
    current_column_index = np.random.randint(environment_columns)
  return current_row_index, current_column_index


def get_next_action(current_row_index, current_column_index, epsilon):
  
  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
  else: 
    return np.random.randint(8)


def get_next_location(current_row_index, current_column_index, action_index):

  new_row_index = current_row_index
  new_column_index = current_column_index

  if actions[action_index] == 'up' and current_row_index > 0:
    new_row_index -= 1
  elif actions[action_index] == 'right' and current_column_index < environment_columns - 1:
    new_column_index += 1
  elif actions[action_index] == 'down' and current_row_index < environment_rows - 1:
    new_row_index += 1
  elif actions[action_index] == 'left' and current_column_index > 0:
    new_column_index -= 1
  elif actions[action_index] == 'uprcross' and current_row_index > 0 and current_column_index < environment_columns - 1:
    new_row_index -= 1
    new_column_index += 1
  elif actions[action_index] == 'uplcross' and current_row_index > 0 and current_column_index > 0:
    new_row_index -= 1
    new_column_index -= 1
  elif actions[action_index] == 'downrcross' and current_row_index < environment_rows - 1 and current_column_index < environment_columns - 1:
    new_row_index += 1
    new_column_index += 1
  elif actions[action_index] == 'downlcross' and current_row_index < environment_rows - 1 and current_column_index > 0:
    new_row_index += 1
    new_column_index -= 1

  return new_row_index, new_column_index


def get_shortest_path(start_row_index, start_column_index):
  
  if is_terminal_state(start_row_index, start_column_index):
    return []
  else: 
    current_row_index, current_column_index = start_row_index, start_column_index
    shortest_path = []
    shortest_path.append([current_row_index, current_column_index])
    
    while not is_terminal_state(current_row_index, current_column_index):
      
      action_index = get_next_action(current_row_index, current_column_index, 1.)
      current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
      shortest_path.append([current_row_index, current_column_index])
      #for i in range(len(shortest_path)):
        #print(shortest_path[i])
    return shortest_path

   
epsilon = 0.9 
discount_factor = 0.9 
learning_rate = 0.9 


steps =[]
for episode in range(1000):
  current_step_count = 0
  row_index, column_index = get_starting_location()

  while not is_terminal_state(row_index, column_index):
    current_step_count +=1
   
    action_index = get_next_action(row_index, column_index, epsilon)

    old_row_index, old_column_index = row_index, column_index 
    row_index, column_index = get_next_location(row_index, column_index, action_index)
    
    reward = rewards[row_index, column_index]
    old_q_value = q_values[old_row_index, old_column_index, action_index]
    temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

    new_q_value = old_q_value + (learning_rate * temporal_difference)
    q_values[old_row_index, old_column_index, action_index] = new_q_value
  steps.append(current_step_count)
#print("---------------------------------------")
#print(steps)



colormap = colors.ListedColormap(["white","red","black"])
pyplot.figure(figsize=(7,7))
pyplot.imshow(data,cmap=colormap)
pyplot.show()

print('Training complete!')

print(get_shortest_path(i_row,i_column))

desired_points = get_shortest_path(i_row,i_column)
#matrix = [[0 for i in range(20)] for j in range(20)]

for point in desired_points:
    x = point[0]
    y = point[1]
    data[x][y] = 2

colormap = colors.ListedColormap(["white","red","blue"])
pyplot.figure(figsize=(7,7))
pyplot.imshow(data,cmap=colormap)
pyplot.show()

table_x= list(range(1,1001)) # [1,2,3...1000] böyle bir x listesi oluşturuyoruz
table_y = steps
pyplot.plot(table_x,table_y)

pyplot.title("Step & Episode Line Graph")
pyplot.xlabel("Episode")
pyplot.ylabel("Step")
pyplot.show()

import matplotlib.pyplot as plt
output = open("trainyval.txt", "r")
#print(output.read())

read_txt = output.read()
train_loss = []
train_acc = []
val_loss = []
val_acc = []
aux = read_txt.split("\n")
for i in range(len(aux)):
    line = aux[i]
    if line[0] != "E":
        subline = line.split(" ")
        train_loss.append(float(subline[7])) 
        train_acc.append(float(subline[10]))        
        val_loss.append(float(subline[13]))
        val_acc.append(float(subline[16]))

plt.title("Accuracy")
plt.plot(train_acc, label= "train_acc")
plt.plot(val_acc,label = "val_acc")
plt.legend()
plt.xlabel("Epochs")
plt.minorticks_on()
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)
plt.savefig('accuracy.png')
plt.cla()

plt.title("Loss")
plt.plot(train_loss, label= "train_loss")
plt.plot(val_loss,label = "val_loss")
plt.legend()
plt.xlabel("Epochs")
plt.minorticks_on()
plt.grid(which='minor', alpha=0.2)
plt.grid(which='major', alpha=0.5)

plt.savefig('loss.png')
# Make container to store saliency values for each property
saliency_m_vir = np.zeros((367, 10), dtype=np.float32)

model.eval()
i = -1
for input, output in test_loader:
    i += 1
    input = input.to(device)
    output = output.to(device)
    
    # Get gradient and send pred through back prop
    input.requires_grad_()
    prediction = model(input)
    prediction.backward()
    
    # Print saliency
    saliency = input.grad.cpu().detach().numpy()
    print(saliency)
    saliency_m_vir[i] = saliency
    
# take abs value of each column and take average saliency for each property
saliency_avg = np.zeros((1,10), dtype=np.float32)
for i in range(10):
    saliency_m_vir[:,i] = np.abs(saliency_m_vir[:,i])
    saliency_avg[:,i]   = np.mean(saliency_m_vir[:,i])
    
print(saliency_avg)

########################################### Visualize the Saliency Values ##########################################
# See which properties contribute most to the neural network's mapping

properties = ['v_max', 'v_rms', 'r_vir', 'scale_radius', 'velocity',
             "J", "spin", "b_to_a", "c_to_a", "t_u"]
colors = ["red", "blue", "green", "magenta", "black",
         "darkorange", "purple", "brown", "yellow", "cyan"]
x = np.arange(len(properties))

i = -1
for x, color, property in zip(x, colors, properties):
    i += 1
    plt.scatter(x, saliency_avg[:,i], c=color, label = property)

plt.legend()
plt.xlabel("Property")
plt.ylabel("Saliency Values")
plt.title("NN: Halo Mass")
plt.savefig("Saliency_m_vir")
plt.show()

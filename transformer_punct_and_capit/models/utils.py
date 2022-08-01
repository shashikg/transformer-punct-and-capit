import torch.nn as nn

class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features, 
        out_features,
        bias=True, 
        dropout=None,
        activation=None,
        device=None, 
        dtype=None
    ):
        super().__init__()
        
        self.dropout = dropout
        self.activation = activation

        self.linear_layer = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout)
            
        if self.activation:
            self.activation_layer = getattr(nn, activation)()

    def forward(self, x):
        if self.dropout:
            x = self.dropout_layer(x)
        
        x = self.linear_layer(x)
        
        if self.activation:
            x = self.activation_layer(x)
            
        return x
      
class ClassificationLayer(nn.Module):
    def __init__(
        self,
        in_features, 
        out_features,
        intermediate_layers,
        bias=True, 
        dropout=None,
        activation=None,
        device=None, 
        dtype=None
    ):
        super().__init__()
    
        if intermediate_layers is not None:
            self.layer_nodes = [in_features] + [int(x.strip()) for x in intermediate_layers.split(',')] + [out_features]
        else:
            self.layer_nodes = [in_features, out_features]

        for i in range(len(self.layer_nodes)-2):
            layer = LinearLayer(self.layer_nodes[i], self.layer_nodes[i+1],
                                bias=bias, dropout=dropout, activation=activation, device=device, dtype=dtype)
            setattr(self, f'layer_{i}', layer)

        layer = LinearLayer(self.layer_nodes[-2], self.layer_nodes[-1],
                            bias=bias, dropout=dropout, activation=None, device=device, dtype=dtype)
        setattr(self, f'layer_{len(self.layer_nodes)-2}', layer)

        self.num_layers = len(self.layer_nodes)-1

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, f'layer_{i}')(x)

        return x
    
def UpperCase(text):
    return text.upper()

def Capitalize(text):
    return text.capitalize()

def Other(text):
    return text

def Comma(text):
    return text+","

def FullStop(text):
    return text+"."

def Question(text):
    return text+"?"

class applyPunct:
    def __init__(self, labels_dict, labels_order):
        self.labels_dict = labels_dict
        self.labels_order = labels_order
        self.label_order_map = {key: val for val, key in enumerate(labels_order)}
        self.label_map = {val: key for key, val in labels_dict.items()}
        self.punct_map = {'O': Other, ',': Comma, '.': FullStop, '?': Question}
        self.capit_map = {'O': Other, 'C': Capitalize, 'U': UpperCase}
        
    def __call__(self, word, pred):
        label = self.label_map[pred].split("|")
        word = self.punct_map[label[self.label_order_map['p']]](word)
        word = self.capit_map[label[self.label_order_map['c']]](word)
        return word
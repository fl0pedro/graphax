from graphax import forward, reverse, cross_country
from graphax.examples import make_simple, make_Helmholtz, make_softmax_attention, make_transformer_encoder
from graphax.transforms.markowitz import minimal_markowitz

edges= make_simple()
order = minimal_markowitz(edges)
print(order)


edges = make_Helmholtz()
order = minimal_markowitz(edges)
print(order)

print(cross_country(order, edges)[1])


edges = make_softmax_attention()
order = minimal_markowitz(edges)
print(order)

print(forward(edges)[1])
print(reverse(edges)[1])
print(cross_country(order, edges)[1])


edges = make_transformer_encoder()
order = minimal_markowitz(edges)
print(order)

print(forward(edges)[1])
print(reverse(edges)[1])
print(cross_country(order, edges)[1])

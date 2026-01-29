# LID-DRIVEN CAVITY PINN MODELING

 Passos do Projeto:
 1 - Realizar simulações CFD mantendo a mesma física com 3 refinamentos de malhas, uma coarse(grosseira) 20x20, uma refinada 40x40, uma super-refinada como referência 80x80
 2 - Visualização dos resultados via Paraview
 3 - Introduzir o pipeline de dados com Machine Learning passando a simulação de malha grosseira (coarse) usando como referência a referência 80x80
 4 - Refinar/corrigir malha com ML fazendo coarse -> superfine 80x80
 5 - Validar métricas RMSE, MAE(erros) e L2(física do modelo)
 6 - Introduzir PINN inicial recebendo dados do modelo ML para comparar ganhos proporcionais em acurácia e na física
 7 - Validar graficamente
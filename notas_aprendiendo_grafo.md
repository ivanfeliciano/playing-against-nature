
### Creencias de conexión p<sub>ij</sub> iniciales, *t*<sub>1</sub>
|Arista|Probabilidad de conexión|
|--|--|
|Enfermedad->Final | 0.95 |
|Enfermedad->Reaccion| 0.1 |
|Reaccion->Final|0.95|
|Tratamiento->Final |0.95|
|Tratamiento->Reaccion | 0.95|

### Observación después de tomar acción

|Variable | Valor |
|--|--|
|Enfermedad| 0|
|**Tratamiento** | 1
|Reaccion | 1
|Final | 1

## Actualizando P<sub>Enfermedad->Final</sub>


### Parámetros para el modelo sin la arista *Enfermedad -> Final*

| Tratamiento(0) | 0.5 |
|----------------|-----|
| Tratamiento(1) | 0.5 |


| Tratamiento | Tratamiento(0) | Tratamiento(1) |
|-------------|----------------|----------------|
| Reaccion(0) | 0.7            | 0.4            |
| Reaccion(1) | 0.3            | 0.6            |



| Reaccion    | Reaccion(0)    | Reaccion(0)         | Reaccion(1)    | Reaccion(1)    |
|-------------|----------------|---------------------|----------------|----------------|
| Tratamiento | Tratamiento(0) | Tratamiento(1)      | Tratamiento(0) | Tratamiento(1) |

| Final(0)    | 0.5            | 0.85  | 0.0            | 0.0            |
|-------------|----------------|---------------------|----------------|----------------|
| Final(1)    | 0.5            | 0.15 | 1.0            | 1.0            |

### Tablas de probabilidad conjunta

#### Grafo completo
| Final    | Enfermedad    | Reaccion    | Tratamiento    |   phi(Final,Enfermedad,Reaccion,Tratamiento) |
|----------|---------------|-------------|----------------|----------------------------------------------|
| Final(0) | Enfermedad(0) | Reaccion(0) | Tratamiento(0) |                                       0.1470 |
| Final(0) | Enfermedad(0) | Reaccion(0) | Tratamiento(1) |                                       0.1120 |
| Final(0) | Enfermedad(0) | Reaccion(1) | Tratamiento(0) |                                       0.0000 |
| Final(0) | Enfermedad(0) | Reaccion(1) | Tratamiento(1) |                                       0.0000 |
| Final(0) | Enfermedad(1) | Reaccion(0) | Tratamiento(0) |                                       0.0420 |
| Final(0) | Enfermedad(1) | Reaccion(0) | Tratamiento(1) |                                       0.0540 |
| Final(0) | Enfermedad(1) | Reaccion(1) | Tratamiento(0) |                                       0.0000 |
| Final(0) | Enfermedad(1) | Reaccion(1) | Tratamiento(1) |                                       0.0000 |
| Final(1) | Enfermedad(0) | Reaccion(0) | Tratamiento(0) |                                       0.0980 |
| Final(1) | Enfermedad(0) | Reaccion(0) | Tratamiento(1) |                                       0.0280 |
| Final(1) | Enfermedad(0) | Reaccion(1) | Tratamiento(0) |                                       0.1050 |
| Final(1) | Enfermedad(0) | Reaccion(1) | Tratamiento(1) |                                       0.2100 |
| Final(1) | Enfermedad(1) | Reaccion(0) | Tratamiento(0) |                                       0.0630 |
| Final(1) | Enfermedad(1) | Reaccion(0) | Tratamiento(1) |                                       0.0060 |
| Final(1) | Enfermedad(1) | Reaccion(1) | Tratamiento(0) |                                       0.0450 |
| Final(1) | Enfermedad(1) | Reaccion(1) | Tratamiento(1) |                                       0.0900 |

#### Con subgrafo sin conexión Enfermedad -> Final (eliminando nodo Enfermedad)

| Tratamiento    | Reaccion    | Final    |   phi(Tratamiento,Reaccion,Final) |
|----------------|-------------|----------|-----------------------------------|
| Tratamiento(0) | Reaccion(0) | Final(0) |                            0.1750 |
| Tratamiento(0) | Reaccion(0) | Final(1) |                            0.1750 |
| Tratamiento(0) | Reaccion(1) | Final(0) |                            0.0000 |
| Tratamiento(0) | Reaccion(1) | Final(1) |                            0.1500 |
| Tratamiento(1) | Reaccion(0) | Final(0) |                            0.1700 |
| Tratamiento(1) | Reaccion(0) | Final(1) |                            0.0300 |
| Tratamiento(1) | Reaccion(1) | Final(0) |                            0.0000 |
| Tratamiento(1) | Reaccion(1) | Final(1) |                            0.3000 |

### Regla de actualización
t<sub>1</sub>
P<sub>Enfermedad->Final</sub> = (0.95 * 0.21) / (0.21 + 0.3) = 0.40
t<sub>2</sub>
P<sub>Enfermedad->Final</sub> = 0.104
.
.
.
P<sub>Enfermedad->Final</sub> = 0

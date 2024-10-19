
#### Documento

https://docs.google.com/document/d/19xAjDHIzZsEnhOWnLjRJVtR1h3HYCzogNxp8l4U9y0I/edit

#### Figma
DER

https://www.figma.com/team_invite/redeem/xcmJg1WWzuhLV9tSYoQz0N

#### MODELO RELACIONAL

https://www.figma.com/board/uqqN2ru6oPb4kR6EC6SkaT/Modelo-Relacional---TP1-Labo-de-Datos?node-id=0-1&t=IaSqsYdIe4Wpj9Vd-1

### TO-DO

- nulls en ISO3 en sedes
- consultas SQL
- flujo migratorio con SQL 
- graficos
- pasar en limpio python
- informe

# NOTAS
- en tabla emigrar en cantidad hay algunos valores q tienen .. (vienen desde la tabla original de migraciones) creo q si los reemplazamos por 0 todo ok (lo comente en pandas al principio del .py de sql nose como pasarlo al jupyter, lo probe con eso y corre), pq no puedo convertirlo a un int para hacer la cuenta en i) e ii)
- hay q cambiar en los nombres de las columnas año por anio no me deja usar la ñ (comente al principio hay q pasarlo a jupyter, q quede desde antes de exportar)
- en tabla sedes todos los que tienen 1 seccion o mas en ISO tienen null, los q tienen 0 secciones no pasa capaz sale de ahi el error nose
- me falto lo de ordenar q pide cada uno me olvide, no estoy muy segura de lo q hice seguro sale mas facil y con menos consultas
HAY Q REVISAR LAS CONSULTAS, PARA EL i) me devuelve solo 24 paises raro, y el resto lo mismo

Per risolvere questo errore:

![TurtleBot3 AutoRace model error](docs/autorace_models_error.png)

AGGIUNGERE UN `model.config` alla cartella `turtlebot3_autorace_2020`:


```
<?xml version="1.0"?>
<model>
  <name>TurtleBot3 AutoRace 2020</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf> <!-- If you have a top-level SDF file -->
  <description>AutoRace 2020 environment for TurtleBot3</description>
</model>
```

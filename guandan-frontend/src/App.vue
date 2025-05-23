<!-- src/App.vue -->
<template>
  <div id="app" style="min-height: 100vh; display: flex; flex-direction: column;">
    <router-view style="
      flex: 1;
      width: 100%;
      display: flex;
      justify-content: center;
      align-items: center;
    "/>
  </div>
</template>

<style>
body {
  margin: 0;
  font-family: sans-serif;
}
</style>

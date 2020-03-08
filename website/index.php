<?php
    require_once 'server.php'
?>
<!DOCTYPE html>
<html lang=en>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spatial</title>
    <link rel="stylesheet" href="default.css">
    <link rel="stylesheet" href="index.css">
    <link href="https://fonts.googleapis.com/css?family=Roboto&display=swap" rel="stylesheet">
    <link rel="shortcut icon" href="favicon.png" type="image/png">
</head>
<body>
    <nav>
        <img src="favicon.png" style="height:3rem; float:left">
        <a style="float:left">Spatial</a>
    </nav>
    <div class="banner">
        <h1 style="line-height: 6rem">View recent Warnings</h1>
        <h4 id="count"><?php echo $_SESSION['count'] ?> Warnings Found</h4>
    </div>
    <div class="main">
        <div id="result">
        </div>
    </div>
    <script src="index.js"></script>
</body>
</html>
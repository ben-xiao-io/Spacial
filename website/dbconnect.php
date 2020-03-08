<?php
    $conn = new mysqli('65.19.141.67', 'xtornado_no', 'newhacks2020', 'xtornado_no');

    if ($conn->connect_error) {
        die("Could not connect: " . $conn->connect_error);
    }

    session_start();
?>
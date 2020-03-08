<?php
    require_once('dbconnect.php');

    $alerts = array();

    $sql = "SELECT image, time
            FROM alerts
            ORDER BY time DESC";

    $result = $conn->query($sql);

    while ($row = $result->fetch_assoc()) {
        $alerts[] = $row;
    }

    $_SESSION['count'] = count($alerts);

    foreach ($alerts as $alert) {
?>
        <div class="alert">
            <h2 style="float:left; line-height: 2.5rem; padding: 0 0.5rem">Collision Warning</h2>
            <span style="float:right; line-height: 2.5rem; padding: 0 0.5rem">
                <?php
                    $daten = new DateTime($alert['time']);
                    echo $daten->format('m/d/Y - g:i A');
                ?>
            </span>
            <hr class="line">
            <img src="<?php echo $alert['image']?>">
        </div>
<?php
    }
?>
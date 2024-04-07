import React from "react";
import { Link } from "react-router-dom";
import './css/Welcome.scss';

class Welcome extends React.Component{

    render() {
        return(
            <section id="main">
                <div className="nav-item">
                    <h1 className="navbar-brand">VIRTUAICOMPOSER</h1>
                </div>

                <div className="main-row">
                    <div className="main-row-text">
                        <h1>Music for everyone</h1>
                        <p>Produce the music you want yourself</p>
                        <Link to="/home" className="btn" style={{textDecoration: "none", backgroundColor:"#9d9063", color:"black"}}>Start</Link>
                    </div>
                </div>
            </section>
        );
    }
}

export default Welcome;

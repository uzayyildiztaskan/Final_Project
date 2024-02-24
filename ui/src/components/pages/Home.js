import React from "react";
import { Link } from "react-router-dom";
import './css/Home.scss';

class Homepage extends React.Component {
    render() {
        return (
            <section id="main">
                <div className="header">

                    <h1 className="navbar-brand">VIRTUAICOMPOSER</h1>
                </div>
                <div className="prompt-container">
                    <h2 style={{color:'#67584d'}}>Ready to create music?</h2>
                    <p style={{color:'#67584d'}}>Tell us what you have in mind, and we'll help you compose it!</p>
                    <textarea placeholder="Please enter what kind of music you want ..." />
                    <Link to="/compose" className="btn" style={{ textDecoration: "none" }}>
                        Compose Now
                    </Link>
                </div>
            </section>
        );
    }
}

export default Homepage;

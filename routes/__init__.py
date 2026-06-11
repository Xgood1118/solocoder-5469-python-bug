from flask import Flask


def register_blueprints(app: Flask):
    from routes.predict import predict_bp
    from routes.tasks import tasks_bp
    from routes.stats import stats_bp
    from routes.feedback import feedback_bp
    from routes.cluster import cluster_bp

    app.register_blueprint(predict_bp)
    app.register_blueprint(tasks_bp)
    app.register_blueprint(stats_bp)
    app.register_blueprint(feedback_bp)
    app.register_blueprint(cluster_bp)

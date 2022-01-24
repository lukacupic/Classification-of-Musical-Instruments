package hr.fer.zemris.diplomski;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.ColorUtils;
import android.util.AttributeSet;

public class PulsatingButton extends FloatingActionButton {

    private float MIN_RADIUS = 0;
    private float MAX_RADIUS = -1;
    private float radius = MIN_RADIUS;
    private boolean animating;

    private Paint circlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private Paint mBackgroundPaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    public PulsatingButton(Context context) {
        super(context);
    }

    public PulsatingButton(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public PulsatingButton(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);

        if (!animating) return;

        MAX_RADIUS = (int) (0.5 * getWidth());
        double ratio = (radius - MIN_RADIUS) / (MAX_RADIUS - MIN_RADIUS);
        int alpha = Math.min((int) (255 * (1 - ratio)), 100);
        int accentColor = ContextCompat.getColor(getContext(), R.color.colorAccentDark);
        accentColor = ColorUtils.setAlphaComponent(accentColor, alpha);

        circlePaint.setColor(accentColor);
        mBackgroundPaint.setColor(accentColor);

        if (radius >= MAX_RADIUS) radius = MIN_RADIUS;

        float w = getMeasuredWidth();
        float h = getMeasuredHeight();

        canvas.drawCircle(w / 2, h / 2, MIN_RADIUS, circlePaint);
        canvas.drawCircle(w / 2, h / 2, radius, mBackgroundPaint);

        radius = (radius + 0.5f);
        invalidate();
    }

    public void animateButton(boolean animating) {
        this.animating = animating;
        radius = MIN_RADIUS;
        invalidate();
    }
}
